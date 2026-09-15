# RAG System

A Retrieval-Augmented Generation (RAG) pipeline in Python: index your own documents into a local vector store, then ask questions answered from that content — via a CLI or a FastAPI web API.

## Stack

- **Embeddings**: `sentence-transformers`
- **Vector store**: `chromadb`
- **Generation**: Anthropic API (`ANTHROPIC_API_KEY`)
- **Ingestion**: PDF (`PyPDF2`) and Word (`python-docx`) documents
- **API**: FastAPI + Uvicorn
- **Local models**: `ollama` support

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY
```

## CLI usage

```bash
python main.py index --dir data/documents   # index documents into the vector store
python main.py query "your question here"   # ask a single question
python main.py interactive                  # interactive Q&A session
python main.py status                       # show indexed chunk count + API key status
python main.py reset                        # clear the vector index
```

## Web API

```bash
uvicorn app:app --reload
```

- `GET /` — health check
- `GET /status` — indexed chunk count, API key configured status
- `POST /query` — `{"question": "...", "top_k": 5}` → answer + sources + retrieved chunks

## Project layout

```
app.py              FastAPI web interface
main.py             CLI entry point
src/
  ingestion.py       document loading/chunking
  embeddings.py      embedding generation
  vector_store.py    ChromaDB wrapper
  retriever.py       similarity search
  generator.py       answer generation
  rag_pipeline.py     ties it all together
```
