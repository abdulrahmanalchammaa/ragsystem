from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "rag_documents"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, records: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        if not records:
            return

        existing = set(self.collection.get(ids=[r["id"] for r in records])["ids"])
        new_records = [r for r in records if r["id"] not in existing]
        new_embeddings = [embeddings[i] for i, r in enumerate(records) if r["id"] not in existing]

        if not new_records:
            print("  All documents already indexed.")
            return

        self.collection.add(
            ids=[r["id"] for r in new_records],
            documents=[r["text"] for r in new_records],
            embeddings=new_embeddings,
            metadatas=[r["metadata"] for r in new_records],
        )
        print(f"  Indexed {len(new_records)} new chunks.")

    def query(self, embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i],  # cosine similarity
            })
        return hits

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        print("  Vector store cleared.")
