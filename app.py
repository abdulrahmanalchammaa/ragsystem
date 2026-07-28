"""
RAG System FastAPI Web Interface

Start with: uvicorn app:app --reload
"""

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    from src.rag_pipeline import RAGPipeline
    pipeline = RAGPipeline()
    yield


app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation API",
    version="1.0.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    retrieved_chunks: Optional[List[dict]] = None


class StatusResponse(BaseModel):
    indexed_chunks: int
    api_key_configured: bool


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "message": "RAG System API is running"}


@app.get("/status", response_model=StatusResponse, tags=["health"])
def status():
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    return StatusResponse(
        indexed_chunks=pipeline.document_count(),
        api_key_configured=bool(api_key and not api_key.startswith("your_")),
    )


@app.post("/index", tags=["indexing"])
def index_documents(directory: str = "data/documents"):
    if not os.path.isdir(directory):
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory}")
    try:
        count = pipeline.index(directory)
        return {"message": "Indexing complete", "chunks_created": count, "total_indexed": pipeline.document_count()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", tags=["indexing"])
async def upload_and_index(files: List[UploadFile] = File(...)):
    upload_dir = "data/documents"
    os.makedirs(upload_dir, exist_ok=True)

    saved = []
    for upload in files:
        dest = os.path.join(upload_dir, upload.filename)
        content = await upload.read()
        with open(dest, "wb") as f:
            f.write(content)
        saved.append(upload.filename)

    count = pipeline.index(upload_dir)
    return {"uploaded": saved, "total_indexed": pipeline.document_count()}


@app.post("/query", response_model=QueryResponse, tags=["query"])
def query(request: QueryRequest):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY not configured")
    try:
        result = pipeline.query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/index", tags=["indexing"])
def reset_index():
    pipeline.reset_index()
    return {"message": "Index cleared"}
