from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.utils import embedding_functions


@dataclass
class MemoryItem:
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


class MemoryStore:
    def __init__(self, persist_dir: str, collection_name: str, embedding_model: str) -> None:
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, text: str, metadata: Optional[Dict[str, Any]] = None, item_id: Optional[str] = None) -> MemoryItem:
        memory_id = item_id or str(uuid.uuid4())
        self.collection.add(documents=[text], metadatas=[metadata or {}], ids=[memory_id])
        return MemoryItem(id=memory_id, text=text, metadata=metadata)

    def query(self, query_text: str, top_k: int = 5) -> List[MemoryItem]:
        results = self.collection.query(query_texts=[query_text], n_results=top_k)
        out: List[MemoryItem] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for idx, doc in enumerate(docs):
            out.append(MemoryItem(id=ids[idx], text=doc, metadata=metas[idx]))
        return out

    def delete(self, item_id: str) -> None:
        self.collection.delete(ids=[item_id])

    def count(self) -> int:
        return self.collection.count()