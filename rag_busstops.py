import os
import faiss
import json
import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Constants
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "llama3.2"
DATA_FILE = "data/BusStopsWAmenities_8035766100189484498.csv"
CHUNKS_PATH = "busstop_chunks.json"
INDEX_PATH = "busstop_index.idx"

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Custom chunking from bus stops CSV
def preprocess_bus_stops():
    print("📂 Reading and processing bus stop CSV...")
    df = pd.read_csv(DATA_FILE)
    chunks = []

    for _, row in df.iterrows():
        try:
            # Construct semantic chunk
            chunk = {
                "objectId": row["OBJECTID"],
                "stopName": str(row["stop_name"]),
                "jurisdiction": str(row["jurisdiction"]),
                "routes": str(row["Routes"]),
                "bikeRacks": int(row["BikeRacks"]) if pd.notna(row["BikeRacks"]) else 0,
                "text": f"Bus stop '{row['stop_name']}' is located in {row['jurisdiction']} and is served by route(s) {row['Routes']}. It has {int(row['BikeRacks']) if pd.notna(row['BikeRacks']) else 0} bike racks."
            }
            chunks.append(chunk)
        except Exception as e:
            continue  # Skip malformed rows

    print(f"✅ Created {len(chunks)} structured chunks.")
    return chunks

# Embed and index
def build_index(chunks):
    print("🔢 Embedding and indexing chunks...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = np.array([embedder.encode([txt])[0] for txt in tqdm(texts)])

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save for reuse
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w") as f:
        json.dump(chunks, f)
    return index, chunks

# Load from disk or build
def load_or_create_index():
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        print("📦 Loading existing index and chunks...")
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "r") as f:
            chunks = json.load(f)
        return index, chunks
    else:
        chunks = preprocess_bus_stops()
        return build_index(chunks)

# Semantic retrieval
def retrieve(query, index, chunks, top_k=100):
    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, top_k)
    return [chunks[i] for i in I[0]]

# Query Ollama with context
def query_llm(question, context_chunks):
    context = "\n".join([c["text"] for c in context_chunks])
    prompt = f"Use the following bus stop context to answer the question:\n\n{context}\n\nQuestion: {question}"

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(OLLAMA_URL, json=payload)
    answer = ""
    for line in response.text.strip().splitlines():
        try:
            data = json.loads(line)
            content = data.get("message", {}).get("content", "")
            if content:
                answer += content
        except:
            continue
    return answer.strip()

# Interactive mode
if __name__ == "__main__":
    index, chunks = load_or_create_index()
    print(f"📈 Vector count: {index.ntotal}")

    while True:
        query = input("\n❓ Ask a question (or type 'exit'): ")
        if query.lower() in ['exit', 'quit']:
            break
        top_chunks = retrieve(query, index, chunks)
        answer = query_llm(query, top_chunks)
        print("\n🤖 Answer:\n", answer)