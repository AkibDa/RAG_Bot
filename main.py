import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGUpsertResult, RAGSearchResult, RAGQueryResult, RAGChunkAndSrc

load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY not found in environment variables. Please set it in your .env file.")

inngest_client = inngest.Inngest(
    app_id="rag_bot",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):
    
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data.get("pdf_path")
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunk=chunks, source=source_id)
    
    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunk
        source_id = chunks_and_src.source
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"text": chunks[i], "source": source_id} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))
    
    chunks_and_src = await ctx.step.run("Load-and-Chunk-PDF", lambda:_load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("Embed-and-Upsert", lambda:_upsert(chunks_and_src), output_type=RAGUpsertResult)
    
    return ingested.model_dump()

@inngest_client.create_function(
    fn_id="RAG: Query",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k=top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])
    
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))
    
    search_results = await ctx.step.run("Vector-Search", lambda: _search(question, top_k))
    
    context_block = "\n\n".join(f"- {c}" for c in search_results.contexts)
    
    answer_obj = await ctx.step.run_ai(
        "generate-answer",
        lambda: ai.GoogleGemini(
            api_key=os.getenv("GEMINI_API_KEY"),
        ).chat_completion(
            model=os.getenv("GENAI_MODEL", "gemini-pro"),
            messages=[
                ai.ChatMessage(
                    role=ai.ChatRole.SYSTEM,
                    content="You are a helpful assistant. Answer the user's question based ONLY on the context provided.",
                ),
                ai.ChatMessage(
                    role=ai.ChatRole.USER,
                    content=(
                        "Use the following context to answer the question.\n\n"
                        f"Context:\n{context_block}\n\n"
                        f"Question: {question}\n\n"
                        "Answer concisely using only the information from the context above."
                    ),
                ),
            ],
            temperature=0.2,
            max_tokens=1024,
        ),
    )

    answer_text = answer_obj.text()

    final_result = RAGQueryResult(
        answer=answer_text,
        contexts=search_results.contexts,
        sources=list(set(search_results.sources)),
    )
    
    return final_result.model_dump()

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])