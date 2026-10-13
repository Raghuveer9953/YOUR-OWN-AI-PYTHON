"""RAG (Retrieval-Augmented Generation) Pipeline and Document Database in Python.

Provides:
1. Text Chunker with configurable word count and overlap
2. Ollama REST API Client (local embeddings and LLM generation)
3. DocumentDB indexing document chunks using HNSW with cosine distance
"""

import json
import threading
import urllib.request
import urllib.error
from typing import List, Dict, Tuple, Optional, Any
from distance import cosine
from vectordb import HNSW, BruteForce, VectorItem


def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    """Splits a document text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = max(1, chunk_words - overlap_words)
    for i in range(0, len(words), step):
        end = min(i + chunk_words, len(words))
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == len(words):
            break
    return chunks


# =====================================================================
#  OLLAMA CLIENT — Local AI Interface
# =====================================================================

class OllamaClient:
    """HTTP client interacting with a local Ollama instance (default 127.0.0.1:11434)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.host = host
        self.port = port
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"
        self.base_url = f"http://{host}:{port}"

    def is_available(self) -> bool:
        """Checks whether the Ollama server is running and accessible."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except Exception:
            return False

    def embed(self, text: str) -> List[float]:
        """Converts text into high-dimensional vector embedding via Ollama."""
        url = f"{self.base_url}/api/embeddings"
        payload = json.dumps({"model": self.embed_model, "prompt": text}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("embedding", [])
        except Exception:
            return []
        return []

    def generate(self, prompt: str) -> str:
        """Generates LLM response from the local model."""
        url = f"{self.base_url}/api/generate"
        payload = json.dumps({
            "model": self.gen_model,
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("response", "")
        except Exception:
            return "ERROR: Ollama unavailable. Run: ollama serve"
        return "ERROR: Ollama unavailable. Run: ollama serve"


# =====================================================================
#  DOCUMENT DATABASE
# =====================================================================

class DocItem:
    """Represents an indexed document chunk."""

    def __init__(self, doc_id: int, title: str, text: str, emb: List[float]):
        self.id = doc_id
        self.title = title
        self.text = text
        self.emb = list(emb)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "text": self.text,
            "embedding": self.emb,
        }


class DocumentDB:
    """Vector database specifically for chunked text documents and RAG queries."""

    def __init__(self):
        self.store: Dict[int, DocItem] = {}
        self.hnsw = HNSW(m=16, ef_build=200)
        self.bf = BruteForce()
        self.mu = threading.Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        """Stores a chunk and indexes its embedding in HNSW and BruteForce."""
        with self.mu:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            self.store[item.id] = item

            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(
        self, q: List[float], k: int, max_dist: float = 0.7
    ) -> List[Tuple[float, DocItem]]:
        """Semantic search returning top-k chunks with cosine distance <= max_dist."""
        with self.mu:
            if not self.store:
                return []
            # Brute force fallback for small collections (< 10)
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            else:
                raw = self.hnsw.knn(q, k, ef=50, dist=cosine)

            out = []
            for d, item_id in raw:
                if item_id in self.store and d <= max_dist:
                    out.append((d, self.store[item_id]))
            return out

    def remove(self, item_id: int) -> bool:
        with self.mu:
            if item_id not in self.store:
                return False
            del self.store[item_id]
            self.hnsw.remove(item_id)
            self.bf.remove(item_id)
            return True

    def all(self) -> List[DocItem]:
        with self.mu:
            return list(self.store.values())

    def size(self) -> int:
        with self.mu:
            return len(self.store)

    def get_dims(self) -> int:
        with self.mu:
            return self.dims
