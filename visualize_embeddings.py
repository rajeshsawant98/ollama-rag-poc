import json
import faiss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from umap import UMAP
import scipy.sparse

# Load data
CHUNKS_PATH = "chunks.json"
INDEX_PATH = "faiss_index.idx"

with open(CHUNKS_PATH, "r") as f:
    chunks = json.load(f)

index = faiss.read_index(INDEX_PATH)

# Extract embeddings from index
num_vectors = index.ntotal
dimension = index.d
embeddings = np.empty((num_vectors, dimension), dtype="float32")
index.reconstruct_n(0, num_vectors, embeddings)

# Reduce dimensions
print("Reducing dimensions with UMAP...")
umap_model = UMAP(n_components=2, random_state=42)
reduced = umap_model.fit_transform(embeddings)

reduced = np.array(reduced)
print(f"Reduced shape: {reduced.shape}, type: {type(reduced)}")

# Plot
plt.figure(figsize=(12, 8))
plt.scatter(reduced[:, 0], reduced[:, 1], s=10, alpha=0.7, c='purple')

# Optional: annotate a few points
for i, txt in enumerate(chunks[:10]):
    plt.annotate(txt[:30] + "...", (reduced[i, 0], reduced[i, 1]), fontsize=8)

plt.title("2D Visualization of RAG Text Embeddings")
plt.xlabel("UMAP Dim 1")
plt.ylabel("UMAP Dim 2")
plt.grid(True)
plt.show()