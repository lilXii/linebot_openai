import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List

import openai


class RAGStore:
    def __init__(self, knowledge_path: str, embedding_model: str):
        self.knowledge_path = Path(knowledge_path)
        self.embedding_model = embedding_model
        self.documents: List[Dict[str, str]] = []
        self.embeddings: List[List[float]] = []
        self.cache_path = Path(f"{self.knowledge_path}.embeddings.json")
        self._load_knowledge()
        if self.documents:
            if not self._load_cached_embeddings():
                self._embed_documents()
                self._save_cache()

    def _load_knowledge(self) -> None:
        if not self.knowledge_path.exists():
            print(f"[RAG] knowledge base not found: {self.knowledge_path}")
            return
        suffix = self.knowledge_path.suffix.lower()
        if suffix == ".jsonl":
            self._load_jsonl()
        else:
            self._load_plain()
        print(f"[RAG] loaded {len(self.documents)} chunks from {self.knowledge_path}")

    def _load_plain(self) -> None:
        text = self.knowledge_path.read_text(encoding="utf-8", errors="ignore").strip()
        chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
        for idx, chunk in enumerate(chunks):
            self.documents.append(
                {
                    "text": chunk,
                    "doc_path": str(self.knowledge_path.name),
                    "chunk_id": idx,
                    "doc_hash": self._hash_text(chunk),
                }
            )

    def _load_jsonl(self) -> None:
        with self.knowledge_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text = record.get("text", "").strip()
                    if not text:
                        continue
                    self.documents.append(
                        {
                            "text": text,
                            "doc_path": record.get("doc_path", self.knowledge_path.name),
                            "chunk_id": record.get("chunk_id"),
                            "doc_hash": self._hash_text(text),
                        }
                    )
                except json.JSONDecodeError:
                    continue

    def _load_cached_embeddings(self) -> bool:
        if not self.cache_path.exists():
            print(f"[RAG] cache not found: {self.cache_path}")
            return False
        try:
            cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[RAG] cache corrupted: {self.cache_path}")
            return False
        if cache.get("model") != self.embedding_model:
            print("[RAG] cache model mismatch, rebuilding embeddings")
            return False
        entries = cache.get("documents", [])
        embedding_map = {entry.get("doc_hash"): entry.get("embedding") for entry in entries}
        embeddings: List[List[float]] = []
        for doc in self.documents:
            embedding = embedding_map.get(doc.get("doc_hash"))
            if embedding is None:
                print("[RAG] cache missing chunks, rebuilding embeddings")
                return False
            embeddings.append(embedding)
        self.embeddings = embeddings
        print(f"[RAG] loaded cached embeddings from {self.cache_path}")
        return True

    def _embed(self, texts: List[str]) -> List[List[float]]:
        response = openai.Embedding.create(model=self.embedding_model, input=texts)
        sorted_data = sorted(response["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in sorted_data]

    def _embed_documents(self) -> None:
        print(f"[RAG] building embeddings with model {self.embedding_model}")
        texts = [doc["text"] for doc in self.documents]
        self.embeddings = self._embed(texts)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, str]]:
        if not self.embeddings:
            return []
        query_embedding = self._embed([query])[0]
        scored = []
        for doc, doc_embedding in zip(self.documents, self.embeddings):
            score = self._cosine_similarity(query_embedding, doc_embedding)
            scored.append((score, doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [doc for _, doc in scored[:top_k]]

    @staticmethod
    def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _save_cache(self) -> None:
        data = {
            "model": self.embedding_model,
            "documents": [
                {"doc_hash": doc.get("doc_hash"), "embedding": emb}
                for doc, emb in zip(self.documents, self.embeddings)
            ],
        }
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(data), encoding="utf-8")
            print(f"[RAG] cached embeddings -> {self.cache_path}")
        except Exception as exc:
            print(f"[RAG] failed to write cache: {exc}")
