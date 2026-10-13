"""Comprehensive test suite for VectorDB and RAG Python implementation.
"""

import unittest
import io
import json
from unittest.mock import MagicMock

from distance import euclidean, cosine, manhattan
from vectordb import VectorItem, BruteForce, KDTree, HNSW, VectorDB
from rag import chunk_text, DocumentDB, OllamaClient
from demo_data import DEMO_ITEMS
from main import VectorDBHandler, load_demo_data


class TestDistanceMetrics(unittest.TestCase):
    def test_euclidean(self):
        a = [0.0, 0.0]
        b = [3.0, 4.0]
        self.assertAlmostEqual(euclidean(a, b), 5.0)

    def test_cosine(self):
        a = [1.0, 0.0]
        b = [1.0, 0.0]
        self.assertAlmostEqual(cosine(a, b), 0.0)

        c = [0.0, 1.0]
        self.assertAlmostEqual(cosine(a, c), 1.0)

        # Identical vectors of different magnitudes
        d = [2.0, 0.0]
        self.assertAlmostEqual(cosine(a, d), 0.0)

    def test_manhattan(self):
        a = [1.0, 2.0]
        b = [4.0, 6.0]
        self.assertAlmostEqual(manhattan(a, b), 7.0)


class TestAlgorithms(unittest.TestCase):
    def setUp(self):
        self.items = [
            VectorItem(1, "Item 1", "catA", [0.1, 0.2, 0.9]),
            VectorItem(2, "Item 2", "catA", [0.15, 0.22, 0.88]),
            VectorItem(3, "Item 3", "catB", [0.9, 0.8, 0.1]),
            VectorItem(4, "Item 4", "catB", [0.85, 0.82, 0.15]),
            VectorItem(5, "Item 5", "catC", [0.5, 0.5, 0.5]),
        ]

    def test_brute_force(self):
        bf = BruteForce()
        for item in self.items:
            bf.insert(item)

        q = [0.1, 0.2, 0.9]
        res = bf.knn(q, k=2, dist=cosine)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0][1], 1)  # Exact match should be closest

        bf.remove(1)
        res2 = bf.knn(q, k=2, dist=cosine)
        self.assertEqual(res2[0][1], 2)

    def test_kdtree(self):
        kdt = KDTree(dims=3)
        for item in self.items:
            kdt.insert(item)

        q = [0.1, 0.2, 0.9]
        res = kdt.knn(q, k=2, dist=euclidean)
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0][1], 1)

        # Test rebuild
        rem = [it for it in self.items if it.id != 1]
        kdt.rebuild(rem)
        res2 = kdt.knn(q, k=2, dist=euclidean)
        self.assertEqual(res2[0][1], 2)

    def test_hnsw(self):
        hnsw = HNSW(m=4, ef_build=50, seed=42)
        for item in self.items:
            hnsw.insert(item, dist=cosine)

        q = [0.1, 0.2, 0.9]
        res = hnsw.knn(q, k=2, ef=20, dist=cosine)
        self.assertGreaterEqual(len(res), 1)
        self.assertEqual(res[0][1], 1)

        info = hnsw.get_info()
        self.assertEqual(info["nodeCount"], 5)
        self.assertGreaterEqual(info["topLayer"], 0)

        hnsw.remove(1)
        self.assertEqual(len(hnsw), 4)


class TestVectorDB(unittest.TestCase):
    def test_vector_db_workflow(self):
        db = VectorDB(dims=16)
        load_demo_data(db)
        self.assertEqual(db.size(), 20)

        # Test search with all 3 algorithms
        sample_q = DEMO_ITEMS[0]["embedding"]
        for algo in ["bruteforce", "kdtree", "hnsw"]:
            for metric in ["cosine", "euclidean", "manhattan"]:
                res = db.search(sample_q, k=3, metric=metric, algo=algo)
                self.assertEqual(len(res["results"]), 3)
                self.assertIn("latencyUs", res)
                self.assertEqual(res["results"][0]["id"], 1)

        # Test benchmark
        bench = db.benchmark(sample_q, k=3, metric="cosine")
        self.assertEqual(bench["itemCount"], 20)
        self.assertIn("bruteforceUs", bench)
        self.assertIn("kdtreeUs", bench)
        self.assertIn("hnswUs", bench)

        # Test insert and remove
        new_id = db.insert("Test item", "test", [0.5] * 16)
        self.assertEqual(db.size(), 21)
        self.assertTrue(db.remove(new_id))
        self.assertEqual(db.size(), 20)
        self.assertFalse(db.remove(9999))


class TestRAGPipeline(unittest.TestCase):
    def test_chunking(self):
        text = "word " * 600
        chunks = chunk_text(text, chunk_words=250, overlap_words=30)
        self.assertGreater(len(chunks), 1)

        short_text = "hello world this is short"
        chunks_short = chunk_text(short_text, chunk_words=50)
        self.assertEqual(len(chunks_short), 1)
        self.assertEqual(chunks_short[0], short_text)

    def test_document_db(self):
        doc_db = DocumentDB()
        doc_id = doc_db.insert("Doc 1", "This is some content", [0.1, 0.2, 0.3])
        self.assertEqual(doc_id, 1)
        self.assertEqual(doc_db.size(), 1)
        self.assertEqual(doc_db.get_dims(), 3)

        hits = doc_db.search([0.1, 0.2, 0.3], k=1, max_dist=0.7)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][1].title, "Doc 1")


class TestHTTPHandler(unittest.TestCase):
    def setUp(self):
        self.db = VectorDB(dims=16)
        load_demo_data(self.db)
        self.doc_db = DocumentDB()
        self.ollama = OllamaClient()

        # Mock Ollama methods to test document and RAG flow reliably
        self.ollama.is_available = MagicMock(return_value=True)
        self.ollama.embed = MagicMock(return_value=[0.1] * 768)
        self.ollama.generate = MagicMock(return_value="This is a generated test answer.")

        VectorDBHandler.db = self.db
        VectorDBHandler.doc_db = self.doc_db
        VectorDBHandler.ollama = self.ollama
        VectorDBHandler.web_dir = "/Users/raghuveerraja11/.gemini/antigravity/scratch/Your-OWN-AI-Python"

    def _execute_request(self, method: str, path: str, body: dict = None) -> tuple:
        raw_body = json.dumps(body).encode("utf-8") if body else b""
        request_text = f"{method} {path} HTTP/1.1\r\n"
        request_text += f"Host: localhost\r\n"
        if raw_body:
            request_text += f"Content-Type: application/json\r\n"
            request_text += f"Content-Length: {len(raw_body)}\r\n"
        request_text += "\r\n"
        raw_request = request_text.encode("utf-8") + raw_body

        class MockReqHandler(VectorDBHandler):
            def __init__(self, raw):
                self.rfile = io.BytesIO(raw)
                self.wfile = io.BytesIO()
                self.requestline = ""
                self.raw_requestline = b""
                self.handle_one_request()

        handler = MockReqHandler(raw_request)
        response_bytes = handler.wfile.getvalue()

        # Parse response
        header_part, _, body_part = response_bytes.partition(b"\r\n\r\n")
        header_lines = header_part.decode("utf-8").split("\r\n")
        status_line = header_lines[0]
        status_code = int(status_line.split(" ")[1])

        parsed_json = None
        if body_part:
            try:
                parsed_json = json.loads(body_part.decode("utf-8"))
            except Exception:
                pass

        return status_code, parsed_json, body_part

    def test_status_endpoint(self):
        code, data, _ = self._execute_request("GET", "/status")
        self.assertEqual(code, 200)
        self.assertEqual(data["demoCount"], 20)
        self.assertEqual(data["demoDims"], 16)
        self.assertTrue(data["ollamaAvailable"])

    def test_stats_endpoint(self):
        code, data, _ = self._execute_request("GET", "/stats")
        self.assertEqual(code, 200)
        self.assertEqual(data["count"], 20)
        self.assertIn("hnsw", data["algorithms"])

    def test_items_endpoint(self):
        code, data, _ = self._execute_request("GET", "/items")
        self.assertEqual(code, 200)
        self.assertEqual(len(data), 20)

    def test_search_endpoint(self):
        v_str = ",".join(str(x) for x in DEMO_ITEMS[0]["embedding"])
        code, data, _ = self._execute_request(
            "GET", f"/search?v={v_str}&k=3&metric=cosine&algo=hnsw"
        )
        self.assertEqual(code, 200)
        self.assertEqual(len(data["results"]), 3)
        self.assertEqual(data["results"][0]["id"], 1)

    def test_benchmark_endpoint(self):
        v_str = ",".join(str(x) for x in DEMO_ITEMS[0]["embedding"])
        code, data, _ = self._execute_request("GET", f"/benchmark?v={v_str}&k=5&metric=cosine")
        self.assertEqual(code, 200)
        self.assertIn("hnswUs", data)
        self.assertEqual(data["itemCount"], 20)

    def test_hnsw_info_endpoint(self):
        code, data, _ = self._execute_request("GET", "/hnsw-info")
        self.assertEqual(code, 200)
        self.assertEqual(data["nodeCount"], 20)
        self.assertIn("nodesPerLayer", data)
        self.assertIn("edges", data)

    def test_insert_and_delete_endpoint(self):
        insert_body = {
            "metadata": "Custom CS Node",
            "category": "cs",
            "embedding": [0.1] * 16,
        }
        code, data, _ = self._execute_request("POST", "/insert", insert_body)
        self.assertEqual(code, 200)
        new_id = data["id"]

        del_code, del_data, _ = self._execute_request("DELETE", f"/delete/{new_id}")
        self.assertEqual(del_code, 200)
        self.assertTrue(del_data["ok"])

    def test_document_and_rag_endpoints(self):
        # 1. Insert document
        doc_payload = {
            "title": "Quantum Computing",
            "text": "Quantum computers use qubits to perform calculations at unprecedented speeds.",
        }
        code, data, _ = self._execute_request("POST", "/doc/insert", doc_payload)
        self.assertEqual(code, 200)
        self.assertEqual(data["chunks"], 1)
        doc_id = data["ids"][0]

        # 2. List documents
        code, doc_list, _ = self._execute_request("GET", "/doc/list")
        self.assertEqual(code, 200)
        self.assertGreaterEqual(len(doc_list), 1)

        # 3. Search document
        code, search_res, _ = self._execute_request(
            "POST", "/doc/search", {"question": "What are qubits?", "k": 3}
        )
        self.assertEqual(code, 200)
        self.assertIn("contexts", search_res)

        # 4. Ask RAG
        code, ask_res, _ = self._execute_request(
            "POST", "/doc/ask", {"question": "Explain qubits", "k": 3}
        )
        self.assertEqual(code, 200)
        self.assertIn("answer", ask_res)
        self.assertEqual(ask_res["answer"], "This is a generated test answer.")

        # 5. Delete document
        code, del_res, _ = self._execute_request("DELETE", f"/doc/delete/{doc_id}")
        self.assertEqual(code, 200)
        self.assertTrue(del_res["ok"])


if __name__ == "__main__":
    unittest.main()
