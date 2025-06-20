import os
import faiss
import json
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# Constants
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2"
DATA_DIR = "data"
CHUNK_SIZE = 300
INDEX_PATH = "faiss_index.idx"
CHUNKS_PATH = "chunks.json"

# Initialize embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load and chunk data
def load_documents():
    docs = []

    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)
        text_chunks = []

        if filename.startswith("."):  # Skip hidden files like .DS_Store
            continue

        if filename.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                words = text.split()
                text_chunks = [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

        elif filename.endswith(".pdf"):
            reader = PdfReader(path)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            words = text.split()
            text_chunks = [" ".join(words[i:i + CHUNK_SIZE]) for i in range(0, len(words), CHUNK_SIZE)]

        elif filename.endswith(".csv"):
            df = pd.read_csv(path)
            for _, row in df.iterrows():
                fields = []
                for col in df.columns:
                    value = row[col]
                    if pd.notna(value) and str(value).strip() != "":
                        fields.append(f"{col.strip().capitalize()}: {str(value).strip()}")
                if fields:
                    chunk = "\n".join(fields)
                    text_chunks.append(chunk)

        docs.extend(text_chunks)

    return docs

# Build FAISS index
def build_index(chunks):
    print(f"📄 Total chunks to embed: {len(chunks)}")
    embeddings = []

    for chunk in tqdm(chunks, desc="🔢 Embedding chunks"):
        emb = embedder.encode([chunk])[0]
        embeddings.append(emb)

    embeddings = np.array(embeddings)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings, chunks

# Retrieve relevant chunks
def retrieve(query, index, chunks, top_k=1000):
    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, top_k)
    return [chunks[i] for i in I[0]]

# Query Ollama
def query_ollama(question, context):
    prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post(OLLAMA_URL, json=payload)
    lines = response.text.strip().splitlines()
    answer = ""

    for line in lines:
        try:
            data = json.loads(line)
            content = data.get("message", {}).get("content", "")
            if content:
                answer += content
        except:
            continue

    return answer.strip()

# Load or create index
def load_or_create_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        print("📦 Loading existing FAISS index and chunks...")
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "r") as f:
            chunks = json.load(f)
        return index, None, chunks
    else:
        print("🔍 Indexing documents...")
        chunks = load_documents()
        index, embeddings, chunks = build_index(chunks)
        faiss.write_index(index, INDEX_PATH)
        with open(CHUNKS_PATH, "w") as f:
            json.dump(chunks, f)
        return index, embeddings, chunks

# Main interaction loop
if __name__ == "__main__":
    index, embeddings, chunks = load_or_create_index()

    print(f"📈 Vector count: {index.ntotal}")
    if embeddings is not None:
        print(f"🧠 Embedding shape: {embeddings.shape}")

    while True:
        question = input("\n❓ Ask a question (or type 'exit'): ")
        if question.lower() in ["exit", "quit"]:
            break

        top_chunks = retrieve(question, index, chunks)
        context = "\n\n".join(top_chunks)
        answer = query_ollama(question, context)

        print("\n🤖 Answer:\n", answer)