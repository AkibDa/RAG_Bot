# Gemini-Powered RAG Pipeline with Inngest and Qdrant
This project implements a complete Retrieval-Augmented Generation (RAG) pipeline to chat with your PDF documents. It uses Google's Gemini models for embeddings and text generation, Qdrant as a vector database, Inngest for orchestrating the background jobs, and Streamlit for the user interface.

## Core Technologies
**Orchestration: Inngest -** Manages the background tasks for document ingestion and question-answering, providing resilience and observability.

**Generative AI & Embeddings: Google Gemini (gemini-pro, embedding-001) -** Used for generating answers and creating vector embeddings from text chunks.

**Vector Database: Qdrant -** Stores the vector embeddings of the PDF content for efficient similarity search.

**Web UI: Streamlit -** Provides a simple, interactive web interface for uploading PDFs and asking questions.

**API Framework: FastAPI -** Serves the Inngest functions.

**PDF Parsing: LlamaIndex (PDFReader) -** Handles the extraction and chunking of text from PDF documents.

## How It Works
The project is divided into two main asynchronous workflows orchestrated by Inngest.

### 1. Ingestion Flow
When a user uploads a PDF via the Streamlit UI:

The PDF is saved locally.

The Streamlit app sends an event (rag/ingest_pdf) to Inngest.

An Inngest function (rag_ingest_pdf) is triggered, which executes a series of steps:
a.  Load & Chunk: The PDF is loaded, and its text is split into smaller, manageable chunks.
b.  Embed & Upsert: Each text chunk is converted into a vector embedding using the Gemini embedding model. These embeddings, along with their corresponding text and metadata, are then "upserted" into the Qdrant vector database.

### 2. Query Flow
When a user asks a question in the Streamlit UI:

The Streamlit app sends an event (rag/query_pdf_ai) with the question to Inngest.

An Inngest function (rag_query_pdf_ai) is triggered:
a.  Embed Query: The user's question is converted into a vector embedding using the same Gemini model.
b.  Vector Search: This query embedding is used to search Qdrant for the most semantically similar text chunks from the ingested documents.
c.  Generate Answer: The retrieved text chunks (the "context") and the original question are passed to the Gemini generative model.
d.  Return Result: The model generates a concise answer based only on the provided context. This answer, along with its source documents, is returned as the function's output.

The Streamlit UI polls an Inngest API endpoint for the function's result and displays the answer and sources once the run is complete.

## Project Structure
.
├── qdrant_storage/              # Directory for storing uploaded PDFs
├── main.py               # FastAPI server hosting the Inngest functions
├── streamlit_app.py      # The Streamlit user interface
├── data_loader.py        # Handles PDF loading, chunking, and embedding
├── vector_db.py          # Abstraction for interacting with Qdrant
├── custom_types.py       # Pydantic models for structured data
├── .env                  # Your local environment variables (based on .env.example)
├── requirements.txt      # Python dependencies
└── README.md             # This file

## Setup and Installation
### Prerequisites
* *Python 3.9+*

* *Docker and Docker Compose*

* *An active Google AI Studio API Key.*

## Step-by-Step Guide
Clone the Repository
```
git clone <repository-url>
cd <repository-directory>
```
Create a Virtual Environment
```
python -m venv venv
source venv/bin/activate
# On Windows: venv\Scripts\activate
```
Install Dependencies
```
pip install -r requirements.txt

Configure Environment Variables

Create a .env file by copying the example:

cp .env.example .env

Open the .env file and add your Google API Key:

GOOGLE_API_KEY="your-google-api-key-here"
```
## How to Run
You will need to run four separate processes in four different terminal tabs.

### Start Qdrant

The easiest way is with Docker. This command will download the image and start a container.
```
docker run -p 6333:6333 qdrant/qdrant
```

### Start the Inngest Dev Server

The Inngest CLI provides a local development server that mimics the Inngest cloud platform.
```
inngest-cli dev
```
You can view the developer UI at http://127.0.0.1:8288.

### Start the FastAPI App

This serves your Inngest functions so the dev server can communicate with them.
```
uvicorn main:app --reload
```

### Run the Streamlit UI

This starts the user-facing web application.
```
streamlit run streamlit_app.py
```
Open the URL provided by Streamlit (usually http://localhost:8501) in your browser to start using the application.

## Future Improvements
Support for More File Types: Extend data_loader.py to handle .txt, .docx, and other common file formats.

**Conversation History:** Implement memory to allow for follow-up questions.

**Error Handling:** Add more robust error handling for failed API calls or document processing issues.

**Deployment:** Create deployment scripts for services like Streamlit Community Cloud, Google Cloud Run, or AWS ECS.