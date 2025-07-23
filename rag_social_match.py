import json
import faiss
from sentence_transformers import SentenceTransformer

# Load data
with open("dummy_users.json") as f:
    users = json.load(f)
with open("events_with_rsvp_semantic.json") as f:
    events = json.load(f)

# Build user event attendance map
user_events = {}
event_lookup = {}
for event in events:
    eid = event.get("id", event.get("title", ""))
    event_lookup[eid] = event
    for rsvp in event.get("rsvpList", []):
        email = rsvp.get("email")
        if rsvp.get("status") == "attended":
            user_events.setdefault(email, []).append(event.get("title"))

# Build user profiles
embedder = SentenceTransformer("all-MiniLM-L6-v2")
user_chunks = []
user_emails = []
user_metadata = []

for user in users:
    email = user.get("email")
    bio = user.get("bio", "")
    interests = ", ".join(user.get("interests", []))
    loc = user.get("location", {})
    city = loc.get("city", "")
    state = loc.get("state", "")
    attended_titles = user_events.get(email, [])
    attended = ", ".join([title for title in attended_titles[:5] if title]) if attended_titles else "None"

    text = f"{bio}. Interests: {interests}. Location: {city}, {state}. Attended events: {attended}."
    user_chunks.append(text)
    user_emails.append(email)
    user_metadata.append({
        "email": email,
        "city": city,
        "state": state,
        "attended": attended_titles
    })

# Embed and index
embeddings = embedder.encode(user_chunks, show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Query loop
print("🔍 Ready. Ask a natural language question (or type 'exit'):")
while True:
    query = input("\n🧠 Your query: ").strip()
    if query.lower() in ["exit", "quit"]:
        break

    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, 10)

    print("\n👥 Top Matches:")
    for rank, idx in enumerate(I[0]):
        meta = user_metadata[idx]
        score = round(1 / (1 + D[0][rank]), 4)
        attended_list = [title for title in meta['attended'] if title]
        print(f"{rank+1}. {meta['email']} ({meta['city']}) - Score: {score}")
        print(f"   Attended: {', '.join(attended_list) if attended_list else 'None'}\n")