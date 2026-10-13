# YOUR-OWN-AI: Vector Database & Local RAG Engine from Scratch in Python 🚀

A high-performance, lightweight **Vector Database** and **Retrieval-Augmented Generation (RAG) Engine** built completely from scratch in pure Python.

It implements production-grade vector search algorithms side-by-side (**HNSW**, **KD-Tree**, and **Brute Force**), distance metrics, dynamic PCA dimensionality reduction, and an end-to-end RAG question-answering pipeline with a live interactive Web UI.

> **Zero External Dependencies:** Built 100% using the Python Standard Library (`http.server`, `urllib`, `heapq`, `math`, `random`, `threading`). No heavy third-party packages or complex setup required.

---

## 🌟 Key Features

| Feature | Description |
|---|---|
| 🧠 **3 Vector Search Algorithms** | **HNSW** (Hierarchical Navigable Small World), **KD-Tree**, and **Brute Force** search side-by-side |
| 📏 **3 Distance Metrics** | **Cosine Distance** (1 - cos), **Euclidean Distance** (L2), and **Manhattan Distance** (L1) |
| 📊 **2D PCA Semantic Space** | Interactive 2D scatter plot visualizing high-dimensional embeddings and query clusters in real time |
| ⚡ **Live Microsecond Benchmarks** | Direct latency comparisons between HNSW, KD-Tree, and Brute Force |
| 🕸️ **HNSW Graph Topology Inspector** | Live visual breakdown of multilayer graph hierarchy (layers, nodes, edges) |
| 📄 **Document Ingestion & Chunking** | Smart sliding-window word chunker (250 words / 30 words overlap) |
| 🤖 **Local Private RAG Pipeline** | Connects to local Ollama (`nomic-embed-text` & `llama3.2`) for private document Q&A |
| 🔌 **Multi-Threaded REST API** | Full CRUD endpoints: vector search, document indexing, topology inspection, and Q&A |
| 💻 **Cross-Platform** | Works out of the box on Windows, macOS, and Linux |

---

## 🏗️ Architecture: How It Works
