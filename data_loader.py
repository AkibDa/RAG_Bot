import google.generativeai as genai
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
  raise ValueError("GEMINI_API_KEY not found in environment variables.")
genai.configure(api_key=api_key)

EMBED_MODEL = "models/embedding-001"
EMBED_DIM = 3072

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
  docs = PDFReader().load_data(file=path)
  texts = [d.texts for d in docs if getattr(d, 'text', None)]
  chunks = []
  for t in texts:
    chunks.extend(splitter.split_text(t))
  return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
  response = genai.embed_content(
    model=EMBED_MODEL,
    content=texts
  )
  return [item.embedding for item in response.data]
  