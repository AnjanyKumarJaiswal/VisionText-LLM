"""Thin wrapper around a local, on-disk Chroma database."""

from __future__ import annotations

from pathlib import Path

import chromadb


class ChromaVectorDB:
    """Each document lives in its own collection, so documents never mix in a search."""

    def __init__(self, path: Path):
        self.client = chromadb.PersistentClient(path=path)

    def _collection(self, namespace: str) -> chromadb.Collection:
        return self.client.get_or_create_collection(
            name=f"doc-{namespace}",
            configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )

    def count(self, namespace: str) -> int:
        return self._collection(namespace).count()

    def clear(self, namespace: str) -> None:
        self.client.delete_collection(f"doc-{namespace}")

    def upsert(self, vectors: list[dict], namespace: str) -> None:
        collection = self._collection(namespace)
        batch_size = self.client.get_max_batch_size()
        for start in range(0, len(vectors), batch_size):
            batch = vectors[start : start + batch_size]
            collection.upsert(
                ids=[vector["id"] for vector in batch],
                embeddings=[vector["values"] for vector in batch],
                metadatas=[vector["metadata"] for vector in batch],
            )

    def search(self, vector: list[float], namespace: str, kind: str, top_k: int) -> list[dict]:
        """Return the metadata of the `top_k` closest vectors of one kind ("text" or "image")."""
        result = self._collection(namespace).query(
            query_embeddings=[vector],
            n_results=top_k,
            where={"type": kind},
            include=["metadatas"],
        )
        return result["metadatas"][0]
