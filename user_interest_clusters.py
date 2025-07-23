import json
import pandas as pd
import matplotlib.pyplot as plt
from umap import UMAP
from sentence_transformers import SentenceTransformer

# Load users
with open("dummy_users.json") as f:
    users = json.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

user_profiles = []
user_labels = []
user_cities = []

for user in users:
    if isinstance(user.get("interests"), list) and user.get("email"):
        text = f"{user.get('bio', '')} Interests: {', '.join(user['interests'])}"
        user_profiles.append(text)
        user_labels.append(user["email"])
        user_cities.append(user.get("location", {}).get("city", "Unknown"))

# Compute embeddings
embeddings = embedder.encode(user_profiles, show_progress_bar=True)

# Dimensionality reduction
umap = UMAP(n_components=2, random_state=42)
embedding_2d = umap.fit_transform(embeddings)

# Create DataFrame
df = pd.DataFrame({
    "x": embedding_2d[:, 0],
    "y": embedding_2d[:, 1],
    "city": user_cities
})

# Plot
plt.figure(figsize=(12, 8))
for city in df["city"].unique():
    subset = df[df["city"] == city]
    plt.scatter(subset["x"], subset["y"], label=city, alpha=0.6, s=25)

plt.title("User Interest Clusters by City")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="City", loc="best", fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig("user_interest_clusters.png")
plt.show()