from typing import List, Dict, Any, Optional
from src.ingestion import ingest_documents
from src.embeddings import embed_texts
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import generate_answer


class RAGPipeline:
    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        collection_name: str = "rag_documents",
        top_k: int = 5,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_score: float = 0.0,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = VectorStore(persist_dir=persist_dir, collection_name=collection_name)
        self.retriever = Retriever(self.vector_store, top_k=top_k, min_score=min_score)

    def index(self, documents_dir: str) -> int:
        print(f"\nIngesting documents from: {documents_dir}")
        records = ingest_documents(documents_dir, chunk_size=self.chunk_size, overlap=self.chunk_overlap)

        if not records:
            print("  No documents found.")
            return 0

        print(f"  Generated {len(records)} chunks. Embedding...")
        texts = [r["text"] for r in records]
        embeddings = embed_texts(texts)

        print("  Storing in vector database...")
        self.vector_store.add_documents(records, embeddings)
        return len(records)

    def query(self, question: str) -> Dict[str, Any]:
        if self.vector_store.count() == 0:
            return {
                "question": question,
                "answer": "No documents have been indexed yet. Please run indexing first.",
                "sources": [],
            }

        hits = self.retriever.retrieve(question)

        if not hits:
            return {
                "question": question,
                "answer": "No relevant documents found for your question.",
                "sources": [],
            }

        context = self.retriever.format_context(hits)
        answer = generate_answer(question, context)

        sources = list({h["metadata"]["source"] for h in hits})

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": [
                {
                    "text": h["text"][:200] + "..." if len(h["text"]) > 200 else h["text"],
                    "source": h["metadata"]["source"],
                    "score": round(h["score"], 4),
                }
                for h in hits
            ],
        }

    def reset_index(self) -> None:
        self.vector_store.reset()

    def document_count(self) -> int:
        return self.vector_store.count()
