#!/usr/bin/env python3
"""VectorDB Engine & REST API in Python.

Matches the exact endpoints, data formats, and architecture of Your-OWN-AI.
Built completely using Python standard library (zero external dependencies required).
"""

import sys
import os
import json
import time
import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import socket
from urllib.parse import urlparse, parse_qs
from typing import Dict, Any, List

from vectordb import VectorDB
from rag import DocumentDB, OllamaClient, chunk_text
from distance import get_dist_fn
from demo_data import DEMO_ITEMS

DIMS = 16


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server handling concurrent requests with dual-stack IPv4/IPv6 support."""
    daemon_threads = True

    def __init__(self, server_address, RequestHandlerClass):
        host, port = server_address
        if host in ("0.0.0.0", "", "localhost"):
            try:
                self.address_family = socket.AF_INET6
                super().__init__(("", port), RequestHandlerClass, bind_and_activate=False)
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
                self.server_bind()
                self.server_activate()
                return
            except Exception:
                pass
        self.address_family = socket.AF_INET
        super().__init__(server_address, RequestHandlerClass, bind_and_activate=True)


class VectorDBHandler(BaseHTTPRequestHandler):
    """REST API Request Handler for VectorDB and RAG pipeline."""

    db: VectorDB = None
    doc_db: DocumentDB = None
    ollama: OllamaClient = None
    web_dir: str = ""

    def log_message(self, format: str, *args: Any) -> None:
        """Clean custom logging."""
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {args[0]} {args[1]} -> {args[2]}\n")

    def _send_cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length).decode("utf-8")
        try:
            return json.loads(raw)
        except Exception:
            return {}

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._send_cors_headers()
        self.end_headers()

    # =================================================================
    #  GET REQUESTS
    # =================================================================
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        # 1. Root: Serve index.html
        if path == "/" or path == "/index.html":
            html_file = os.path.join(self.web_dir, "index.html")
            if not os.path.exists(html_file):
                self.send_error(404, "index.html not found")
                return
            with open(html_file, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(content)
            return

        # 2. System Status
        if path == "/status":
            ollama_up = self.ollama.is_available()
            self._send_json(200, {
                "ollamaAvailable": ollama_up,
                "embedModel": self.ollama.embed_model,
                "genModel": self.ollama.gen_model,
                "docCount": self.doc_db.size(),
                "docDims": self.doc_db.get_dims(),
                "demoDims": DIMS,
                "demoCount": self.db.size(),
            })
            return

        # 3. Database Statistics
        if path == "/stats":
            self._send_json(200, {
                "count": self.db.size(),
                "dims": DIMS,
                "algorithms": ["bruteforce", "kdtree", "hnsw"],
                "metrics": ["euclidean", "cosine", "manhattan"],
            })
            return

        # 4. List all 16D Demo Items
        if path == "/items":
            self._send_json(200, self.db.all())
            return

        # 5. Search Demo Vectors
        if path == "/search":
            v_param = query.get("v", [""])[0]
            try:
                q = [float(x.strip()) for x in v_param.split(",") if x.strip()]
            except ValueError:
                q = []

            if len(q) != DIMS:
                self._send_json(400, {"error": f"need {DIMS}D vector"})
                return

            try:
                k = int(query.get("k", [5])[0])
            except (ValueError, TypeError):
                k = 5

            metric = query.get("metric", ["cosine"])[0] or "cosine"
            algo = query.get("algo", ["hnsw"])[0] or "hnsw"

            res = self.db.search(q, k, metric, algo)
            self._send_json(200, res)
            return

        # 6. Benchmark Algorithms
        if path == "/benchmark":
            v_param = query.get("v", [""])[0]
            try:
                q = [float(x.strip()) for x in v_param.split(",") if x.strip()]
            except ValueError:
                q = []

            if len(q) != DIMS:
                self._send_json(400, {"error": f"need {DIMS}D vector"})
                return

            try:
                k = int(query.get("k", [5])[0])
            except (ValueError, TypeError):
                k = 5

            metric = query.get("metric", ["cosine"])[0] or "cosine"
            bench = self.db.benchmark(q, k, metric)
            self._send_json(200, bench)
            return

        # 7. HNSW Multi-Layer Graph Topology
        if path == "/hnsw-info":
            self._send_json(200, self.db.hnsw_info())
            return

        # 8. List Indexed Documents
        if path == "/doc/list":
            docs = self.doc_db.all()
            resp = []
            for d in docs:
                preview = d.text[:120]
                if len(d.text) > 120:
                    preview += "…"
                word_count = len(d.text.split())
                resp.append({
                    "id": d.id,
                    "title": d.title,
                    "preview": preview,
                    "words": word_count,
                })
            self._send_json(200, resp)
            return

        self.send_error(404, "Endpoint not found")

    # =================================================================
    #  POST REQUESTS
    # =================================================================
    def do_POST(self) -> None:
        path = urlparse(self.path).path
        body = self._read_json_body()

        # 1. Insert 16D Vector
        if path == "/insert":
            meta = body.get("metadata", "")
            cat = body.get("category", "")
            emb = body.get("embedding", [])

            if not meta or not emb or len(emb) != DIMS:
                self._send_json(400, {"error": "invalid body"})
                return

            new_id = self.db.insert(meta, cat, emb, get_dist_fn("cosine"))
            self._send_json(200, {"id": new_id})
            return

        # 2. Insert Document (RAG chunk & embed)
        if path == "/doc/insert":
            title = body.get("title", "").strip()
            text = body.get("text", "").strip()
            if not title or not text:
                self._send_json(400, {"error": "need title and text"})
                return

            chunks = chunk_text(text, chunk_words=250, overlap_words=30)
            ids = []

            for i, chunk in enumerate(chunks):
                emb = self.ollama.embed(chunk)
                if not emb:
                    self._send_json(503, {
                        "error": (
                            "Ollama unavailable. Install from https://ollama.com then run: "
                            "ollama pull nomic-embed-text && ollama pull llama3.2"
                        )
                    })
                    return

                chunk_title = (
                    f"{title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title
                )
                chunk_id = self.doc_db.insert(chunk_title, chunk, emb)
                ids.append(chunk_id)

            self._send_json(200, {
                "ids": ids,
                "chunks": len(chunks),
                "dims": self.doc_db.get_dims(),
            })
            return

        # 3. Document Search (Visualizer)
        if path == "/doc/search":
            question = body.get("question", "").strip()
            k = int(body.get("k", 3))
            if not question:
                self._send_json(400, {"error": "need question"})
                return

            q_emb = self.ollama.embed(question)
            if not q_emb:
                self._send_json(503, {"error": "Ollama unavailable"})
                return

            hits = self.doc_db.search(q_emb, k)
            contexts = [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "distance": round(d, 4),
                }
                for d, doc in hits
            ]
            self._send_json(200, {"contexts": contexts})
            return

        # 4. RAG Q&A (Embed -> Retrieve -> Prompt -> Generate)
        if path == "/doc/ask":
            question = body.get("question", "").strip()
            k = int(body.get("k", 3))
            if not question:
                self._send_json(400, {"error": "need question"})
                return

            # Step 1: Embed question
            q_emb = self.ollama.embed(question)
            if not q_emb:
                self._send_json(503, {"error": "Ollama unavailable"})
                return

            # Step 2: Retrieve relevant chunks
            hits = self.doc_db.search(q_emb, k)

            # Step 3: Construct Prompt with context
            ctx_str = ""
            for i, (d, doc) in enumerate(hits):
                ctx_str += f"[{i + 1}] {doc.title}:\n{doc.text}\n\n"

            prompt = (
                "You are a helpful assistant. Answer the user's question directly. "
                "Use the provided context if it contains relevant information. "
                "If it doesn't, just use your own general knowledge. "
                "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
                "Just answer the question naturally.\n\n"
                f"Context:\n{ctx_str}"
                f"Question: {question}\n\n"
                "Answer:"
            )

            # Step 4: Generate response via local LLM
            answer = self.ollama.generate(prompt)

            # Step 5: Format response
            contexts = [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "text": doc.text,
                    "distance": round(d, 4),
                }
                for d, doc in hits
            ]

            self._send_json(200, {
                "answer": answer,
                "model": self.ollama.gen_model,
                "contexts": contexts,
                "docCount": self.doc_db.size(),
            })
            return

        self.send_error(404, "Endpoint not found")

    # =================================================================
    #  DELETE REQUESTS
    # =================================================================
    def do_DELETE(self) -> None:
        path = urlparse(self.path).path

        # 1. Delete Demo Item: /delete/<id>
        if path.startswith("/delete/"):
            try:
                item_id = int(path.split("/")[-1])
                ok = self.db.remove(item_id)
                self._send_json(200, {"ok": ok})
            except ValueError:
                self._send_json(400, {"ok": False, "error": "invalid id"})
            return

        # 2. Delete Document Chunk: /doc/delete/<id>
        if path.startswith("/doc/delete/"):
            try:
                doc_id = int(path.split("/")[-1])
                ok = self.doc_db.remove(doc_id)
                self._send_json(200, {"ok": ok})
            except ValueError:
                self._send_json(400, {"ok": False, "error": "invalid id"})
            return

        self.send_error(404, "Endpoint not found")


def load_demo_data(db: VectorDB) -> None:
    """Pre-populates the 16D semantic vectors across 4 categories."""
    dist = get_dist_fn("cosine")
    for item in DEMO_ITEMS:
        db.insert(item["metadata"], item["category"], item["embedding"], dist)


def main():
    parser = argparse.ArgumentParser(description="VectorDB Python Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port number (default: 8080)")
    parser.add_argument("--ollama-host", default="127.0.0.1", help="Ollama host (default: 127.0.0.1)")
    parser.add_argument("--ollama-port", type=int, default=11434, help="Ollama port (default: 11434)")
    args = parser.parse_args()

    # Allow PORT environment variable if set
    port = int(os.environ.get("PORT", args.port))

    # Initialize Engine
    db = VectorDB(dims=DIMS)
    doc_db = DocumentDB()
    ollama = OllamaClient(host=args.ollama_host, port=args.ollama_port)

    # Pre-populate demo vectors
    load_demo_data(db)

    # Check Ollama status
    ollama_up = ollama.is_available()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    VectorDBHandler.db = db
    VectorDBHandler.doc_db = doc_db
    VectorDBHandler.ollama = ollama
    VectorDBHandler.web_dir = script_dir

    print("=" * 45)
    print("=== VectorDB Python Engine ===")
    print(f"http://localhost:{port}")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")
    print("=" * 45)

    server = ThreadedHTTPServer((args.host, port), VectorDBHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down VectorDB server...")
        server.server_close()


if __name__ == "__main__":
    main()
