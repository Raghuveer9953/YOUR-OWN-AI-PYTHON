# VectorDB (Your-OWN-AI) in Python 🚀

> **Original C++ Repository:** [perryvegehan/Your-OWN-AI](https://github.com/perryvegehan/Your-OWN-AI)  
> Yeh wahi same project hai jisko **C++ se Python me 100% convert kiya gaya hai**, bina kisi external library ya complex compiler setup ke!

Ek fully working **Vector Database + RAG (Retrieval-Augmented Generation) Pipeline** jo scratch se Python me implement kiya gaya hai with a modern interactive Web UI.

Pinecone, ChromaDB, aur Weaviate jaise production vector databases under-the-hood kaise kaam karte hain, yeh sikhne aur apna khud ka AI search engine run karne ke liye best educational and practical project hai.

---

## 🌟 Key Features (Kya-Kya Hai Isme?)

| Feature | Description |
|---|---|
| 🧠 **3 Search Algorithms** | **HNSW** (Hierarchical Navigable Small World), **KD-Tree**, aur **Brute Force** side-by-side |
| 📏 **3 Distance Metrics** | Cosine Similarity (1 - cos), Euclidean Distance (L2), aur Manhattan Distance (L1) |
| 📊 **2D PCA Semantic Scatter Plot** | Live visualizer jisme clusters bante hain aur query points glow karte hain |
| ⚡ **Algorithmic Benchmark** | Brute Force vs KD-Tree vs HNSW ki speed comparison (latency in microseconds) |
| 🌐 **Interactive Graph Visualizer** | HNSW multilayer graph ke nodes aur edges ka real-time visual inspection |
| 📄 **Document Ingestion & Chunking** | Long documents ko automatic overlapping word-chunks (250 words / 30 overlap) me split karta hai |
| 🤖 **Local Ollama Integration** | Text embeddings (`nomic-embed-text`) + LLM generation (`llama3.2`) without cloud API keys |
| 💬 **End-to-End RAG Pipeline** | Apne documents paste karo, question pucho → HNSW context retrieve karega → Local LLM accurate answer dega |
| 🔌 **Zero External Dependencies** | 100% Python Standard Library! Koi `pip install` ka jhanjhat nahi! |

---

## 🏗️ Architecture: How It Works

```
           [ User Text / Document ]
                     │
                     ▼
             [ Text Chunker ]               (Overlapping word windows)
                     │
                     ▼
         Ollama (nomic-embed-text)          (Generates 768D Vector Embeddings)
                     │
                     ▼
             [ HNSW Multi-Layer ]            (Fast O(log N) Graph Indexing)
                     │
    ┌────────────────┴────────────────┐
    ▼                                 ▼
[ Semantic Search ]             [ User Question ]
    │                                 │
    ▼                                 ▼
[ Retrieved Context Chunks ] ──► [ RAG Prompt Builder ]
                                      │
                                      ▼
                             Ollama (llama3.2)
                                      │
                                      ▼
                              [ Accurate Answer ]
```

---

## 🛠️ Prerequisites (Kya Chahiye?)

1. **Python 3.8+** (Windows, macOS, ya Linux par)
2. **Ollama** (Local AI models run karne ke liye - Free & Open Source)
   - Download from: [https://ollama.com](https://ollama.com)

> 💡 **C++ version ke mukable fayda:** C++ version me MSYS2, MinGW, g++, Windows PATH setup aur compilation flags lagte the. Python version me **kisi bhi C++ compiler ki zaroorat nahi hai**. Seedha `python main.py` run hota hai!

---

## 🚀 Quick Start Guide (Kaise Run Karein?)

### Step 1: Ollama Setup (Local AI Models)

Ollama install karne ke baad terminal ya command prompt me yeh 2 models pull karein:

```bash
# 1. Embedding Model (~274 MB)
ollama pull nomic-embed-text

# 2. Local LLM (~2.0 GB)
ollama pull llama3.2
```

Ollama background me chalu rahega. Agar na chal raha ho toh run karein:
```bash
ollama serve
```

### Step 2: Run VectorDB Python Server

Terminal me project folder me jayein aur command run karein:

```bash
# Linux / macOS:
python3 main.py

# Windows:
python main.py
# ya run.bat par double click karein
```

Aapko console me output dikhega:
```
=============================================
=== VectorDB Python Engine ===
http://localhost:8080
20 demo vectors | 16 dims | HNSW+KD-Tree+BruteForce
Ollama: ONLINE
  embed model: nomic-embed-text  gen model: llama3.2
=============================================
```

### Step 3: Open in Browser

Apne browser me open karein:
👉 **[http://localhost:8080](http://localhost:8080)**

Yahan aapko:
- 🌌 **2D PCA Semantic Space** dikhega jisme 4 categories (CS, Math, Food, Sports) plotted hain.
- ⚡ **Benchmark tab** me teeno algorithms (Brute Force, KD-Tree, HNSW) ki speed compare kar sakte hain.
- 🕸️ **HNSW Graph tab** me multilayer hierarchy dekh sakte hain.
- 📚 **RAG tab** me koi bhi text/study notes paste karke unse questions puch sakte hain!

---

## 📂 Project Structure

```
Your-OWN-AI-Python/
├── main.py           # Multi-threaded HTTP Server & REST API endpoints
├── vectordb.py       # Core Engine: VectorItem, BruteForce, KDTree, HNSW, VectorDB
├── rag.py            # Text Chunker, OllamaClient & DocumentDB (RAG pipeline)
├── distance.py       # Distance functions: Cosine, Euclidean, Manhattan
├── demo_data.py      # 20 Preloaded 16D semantic vectors across 4 domains
├── index.html        # Interactive Frontend Web UI (Scatter plot, benchmarks, RAG)
├── test_db.py        # Comprehensive Unit Tests (17 unit tests)
├── requirements.txt  # Dependency information (Zero external pip packages needed)
├── run.sh            # Linux / macOS start script
├── run.bat           # Windows start script
└── README.md         # Documentation
```

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI serve karta hai (`index.html`) |
| `GET` | `/status` | Server aur Ollama status (`online`/`offline`, doc count, dims) |
| `GET` | `/stats` | Vector count, dimensions, supported algorithms & metrics |
| `GET` | `/items` | Saare 16D demo vectors fetch karta hai |
| `GET` | `/search?v=...&k=5&metric=cosine&algo=hnsw` | k-Nearest Neighbors search |
| `POST` | `/insert` | Naya vector insert karta hai `{"metadata": "...", "category": "...", "embedding": [...]}` |
| `DELETE` | `/delete/<id>` | ID ke vector ko delete karta hai |
| `GET` | `/benchmark?v=...&k=5&metric=cosine` | BruteForce vs KD-Tree vs HNSW speed benchmark (in microseconds) |
| `GET` | `/hnsw-info` | HNSW graph structure (nodes, edges, layers) return karta hai |
| `POST` | `/doc/insert` | Document chunk karta hai, Ollama se embed karta hai aur store karta hai |
| `GET` | `/doc/list` | Stored document chunks ki list aur preview |
| `DELETE` | `/doc/delete/<id>` | Document chunk delete karta hai |
| `POST` | `/doc/search` | Document vector similarity search |
| `POST` | `/doc/ask` | Full RAG pipeline: question embedding → HNSW retrieval → LLM answer |

---

## 🧪 Running Tests

Saare tests run karne ke liye:

```bash
python3 -m unittest test_db.py
```

Output:
```
Ran 17 tests in 0.015s
OK
```

---

## 💡 C++ vs Python Comparison

| Feature | C++ Implementation | Python Implementation (This Project) |
|---|---|---|
| **Compilation** | Required (`g++ -std=c++17 -O2 ...`) | None (Interpreted, Just Run!) |
| **Tooling Required** | MSYS2 / MinGW / Xcode / make | Standard Python 3.8+ |
| **Portability** | Platform-dependent binaries (.exe) | Platform-independent (Mac, Windows, Linux) |
| **Code Readability** | Low-level pointers, templates | Clean, Pythonic, idiomatic classes |
| **Dependencies** | Requires `httplib.h` & `ws2_32` socket library | 100% Standard Library (`http.server`, `urllib`, `heapq`) |
| **Web UI** | Identical | Identical (100% compatible) |
