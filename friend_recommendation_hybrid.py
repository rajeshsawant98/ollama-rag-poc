import json
import faiss
from sentence_transformers import SentenceTransformer
from geopy.distance import geodesic
from collections import defaultdict

# Load users and events
with open("dummy_users.json") as f:
    users = json.load(f)
with open("events_with_rsvp_semantic.json") as f:
    events = json.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Step 1: Prepare user data
user_profiles = []
user_emails = []
user_locations = {}
user_lookup = {}
user_attendance = defaultdict(set)

for user in users:
    email = user.get("email")
    interests = user.get("interests", [])
    bio = user.get("bio", "")
    location = user.get("location", None)

    if email and interests and isinstance(interests, list):
        profile = f"{bio} Interests: {', '.join(interests)}"
        user_profiles.append(profile)
        user_emails.append(email)
        user_lookup[email] = user
        if isinstance(location, dict):
            lat = location.get("latitude")
            lon = location.get("longitude")
            if lat and lon:
                user_locations[email] = (lat, lon)

# Step 2: Build attendance map
for event in events:
    eid = event.get("id", event.get("title", "unknown"))
    for rsvp in event.get("rsvpList", []):
        if rsvp.get("status") == "attended":
            user_attendance[rsvp["email"]].add(eid)

# Step 3: Generate embeddings
embeddings = embedder.encode(user_profiles, show_progress_bar=True)
faiss_index = faiss.IndexFlatL2(embeddings.shape[1])
faiss_index.add(embeddings)

# Step 4: Recommend friends using hybrid scoring
recommendations = {}
TOP_K = 10

for idx, email in enumerate(user_emails):
    query_emb = embeddings[idx].reshape(1, -1)
    D, I = faiss_index.search(query_emb, TOP_K + 10)  # Extra results for filtering
    recos = []

    base_events = user_attendance[email]
    user_loc = user_locations.get(email)

    for result_idx, i in enumerate(I[0]):
        if i == idx:
            continue  # Skip self

        reco_email = user_emails[i]
        emb_score = float(1 / (1 + D[0][result_idx]))

        # Co-attendance score
        reco_events = user_attendance[reco_email]
        shared_events = base_events.intersection(reco_events)
        co_score = len(shared_events) / len(base_events) if base_events else 0.0

        # Location bonus
        loc_bonus = 0
        reco_loc = user_locations.get(reco_email)
        if user_loc and reco_loc:
            dist_km = geodesic(user_loc, reco_loc).km
            if dist_km <= 10:
                loc_bonus = 0.25
            elif dist_km <= 50:
                loc_bonus = 0.1

        # Final hybrid score
        total_score = 0.5 * emb_score + 0.3 * co_score + 0.2 * loc_bonus
        recos.append({
            "email": reco_email,
            "score": round(total_score, 4),
            "sharedEvents": list(shared_events)
        })

    # Sort and limit top results
    recos = sorted(recos, key=lambda x: -x["score"])
    recommendations[email] = recos[:TOP_K]

# Step 5: Save results
with open("friend_recommendations_hybrid.json", "w") as f:
    json.dump(recommendations, f, indent=2)

print("✅ friend_recommendations_hybrid.json generated using embeddings, co-attendance, and location.")