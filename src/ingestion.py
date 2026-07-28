import os
import re
from pathlib import Path
from typing import List, Dict, Any


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_pdf(path: str) -> str:
    try:
        import PyPDF2
        text_parts = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text_parts.append(page.extract_text() or "")
        return "\n".join(text_parts)
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF support. Run: pip install PyPDF2")


def load_docx(path: str) -> str:
    try:
        from docx import Document
        doc = Document(path)
        return "\n".join(para.text for para in doc.paragraphs)
    except ImportError:
        raise ImportError("python-docx is required for DOCX support. Run: pip install python-docx")


def load_document(path: str) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext in (".docx", ".doc"):
        return load_docx(path)
    elif ext in (".txt", ".md", ".rst", ".csv"):
        return load_text_file(path)
    else:
        return load_text_file(path)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    words = text.split(" ")
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def ingest_documents(directory: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, Any]]:
    docs_path = Path(directory)
    if not docs_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    supported = {".pdf", ".docx", ".doc", ".txt", ".md", ".rst", ".csv"}
    records = []

    for file_path in sorted(docs_path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in supported:
            print(f"  Loading: {file_path.name}")
            try:
                text = load_document(str(file_path))
                chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
                for i, chunk in enumerate(chunks):
                    records.append({
                        "id": f"{file_path.stem}__chunk_{i}",
                        "text": chunk,
                        "metadata": {
                            "source": file_path.name,
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                        },
                    })
            except Exception as e:
                print(f"  Warning: could not process {file_path.name}: {e}")

    return records
