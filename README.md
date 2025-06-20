# 🔠 Ollama RAG PoC – Text, PDF, and Bus Stop CSV Retrieval

This repository contains a **Retrieval-Augmented Generation (RAG)** proof-of-concept powered by **Ollama**, **FAISS**, and **SentenceTransformers**. It supports querying across structured and unstructured data, including:

* 📄 Plaintext files (`.txt`)
* 📘 PDF files (`.pdf`)
* 📊 CSV datasets (e.g., bus stops with amenities)

---

## 💪 Features

* 🔍 Load, chunk, and embed data with `sentence-transformers`
* 💃 Store embeddings in a **FAISS** vector index
* 💬 Use local **Ollama** models (`llama3`, `mistral`, `deepseek-chat`) for intelligent answers
* 📈 Visualize embedding clusters using **UMAP** and **matplotlib**
* 🚌 Specialized RAG pipeline for querying **bus stop datasets** with structured output

---

## 📁 Project Structure

```
rajeshsawant98-ollama-rag-poc/
├── chat.py                  # Basic LLM chat using Ollama
├── rag.py                  # General-purpose RAG for text/pdf/csv
├── rag_busstops.py         # Structured RAG pipeline for bus stops
├── requirements.txt        # Python dependencies
├── visualize/
│   └── visualize_embeddings.py  # UMAP/PCA visualization of vector space
├── data/                   # Add your text/pdf/csv files here
│   ├── example.txt
│   ├── yourfile.pdf
│   └── BusStopsWAmenities_*.csv
```

---

## 🚀 Setup & Installation

### 1. Clone the repo

```bash
git clone https://github.com/rajeshsawant98/ollama-rag-poc.git
cd ollama-rag-poc
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Run Examples

### 🔹 Chat with Ollama

```bash
python chat.py
```

### 🔹 General-purpose RAG (text, PDF, CSV)

```bash
python rag.py
```

### 🔹 Specialized RAG for Bus Stop CSV

```bash
python rag_busstops.py
```

### 🔹 Visualize Embeddings

Make sure `faiss_index.idx` and `chunks.json` exist from a previous run:

```bash
python visualize/visualize_embeddings.py
```

---

## ⚙️ Notes

* Embeddings use: `all-MiniLM-L6-v2`
* Local LLM chat via Ollama (`http://localhost:11434`)
* Modify the model used by editing the `MODEL_NAME` in code (`llama3.2`, `mistral`, etc.)
* Add your files to the `data/` folder for indexing

---

## 📊 Example Query (Bus Stop)

```text
🤔 Ask a question: List all the bus stops in Tempe that have bike racks and are served by the route EART.
🤖 Answer:
Bus stop 'Tempe TC (Bay 6 middle)' is located in Tempe and is served by route(s) EART. It has 2 bike racks.
```

---

## 🧐 Future Ideas

* [ ] Add support for multi-file structured chunking
* [ ] Integrate with Pinecone or ChromaDB for persistent storage
* [ ] Add web UI (e.g., using Streamlit or Next.js frontend)

---

## 👨‍💼 Author

**Rajesh Sawant**
📍 Master's in Software Engineering @ ASU
🔗 [Portfolio (WIP)](https://github.com/rajeshsawant98)
🐍 Python, 🧠 GenAI, 🌐 Full-Stack, 💾 Knowledge Graphs
