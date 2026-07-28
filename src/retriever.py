from typing import List, Dict, Any
from src.embeddings import embed_query
from src.vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, top_k: int = 5, min_score: float = 0.0):
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        query_embedding = embed_query(query)
        hits = self.vector_store.query(query_embedding, n_results=self.top_k)
        return [h for h in hits if h["score"] >= self.min_score]

    def format_context(self, hits: List[Dict[str, Any]]) -> str:
        parts = []
        for i, hit in enumerate(hits, 1):
            source = hit["metadata"].get("source", "unknown")
            parts.append(f"[{i}] Source: {source}\n{hit['text']}")
        return "\n\n---\n\n".join(parts)
