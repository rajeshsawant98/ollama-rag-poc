import json
import faiss
import random
from sentence_transformers import SentenceTransformer

# Load users and events
with open("dummy_users.json") as f:
    users = json.load(f)
with open("events_collection.json") as f:
    events = json.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Build user embeddings and location map
user_profiles, user_emails = [], []
user_locations = {}

for user in users:
    if isinstance(user.get("interests"), list):
        profile = f"{user.get('bio', '')} Interests: {', '.join(user['interests'])}"
        user_profiles.append(profile)
        user_emails.append(user["email"])
        loc = user.get("location", {})
        user_locations[user["email"]] = {
            "city": loc.get("city", "").strip().lower(),
            "state": loc.get("state", "").strip().lower()
        }

user_embeddings = embedder.encode(user_profiles, show_progress_bar=True)
faiss_user_index = faiss.IndexFlatL2(user_embeddings.shape[1])
faiss_user_index.add(user_embeddings)

# Build event embeddings and location map
event_descriptions, event_ids = [], []
event_locations = {}

for event in events:
    desc = event.get("description", "")
    cats = ", ".join(event.get("categories", []) + event.get("tags", []))
    title = event.get("title", "")
    combined = f"{title}\nCategories: {cats}\n{desc}"
    event_descriptions.append(combined)
    eid = event.get("id", title)
    event_ids.append(eid)
    loc = event.get("location", {})
    event_locations[eid] = {
        "city": loc.get("city", "").strip().lower(),
        "state": loc.get("state", "").strip().lower()
    }

event_embeddings = embedder.encode(event_descriptions, show_progress_bar=True)

# Sample reviews
sample_reviews = [
    "Amazing event, would attend again!",
    "Had a great time!",
    "Well organized and fun.",
    "Loved the vibe and energy!",
    "Could have been better.",
    "Not what I expected, but still enjoyable."
]

# Compute RSVPs with location boost
rsvp_map = {}
top_k = 50
D, I = faiss_user_index.search(event_embeddings, top_k)

for event_idx, user_idxs in enumerate(I):
    eid = event_ids[event_idx]
    event_loc = event_locations.get(eid, {})
    scored_users = []

    for rank, user_idx in enumerate(user_idxs):
        email = user_emails[user_idx]
        sim_score = 1 / (1 + D[event_idx][rank])

        # Location boost
        user_loc = user_locations.get(email, {})
        loc_boost = 0
        if user_loc.get("city") == event_loc.get("city") and user_loc["city"]:
            loc_boost = 0.3
        elif user_loc.get("state") == event_loc.get("state") and user_loc["state"]:
            loc_boost = 0.1

        final_score = 0.7 * sim_score + 0.3 * loc_boost
        scored_users.append((final_score, email))

    scored_users.sort(reverse=True)
    rsvp_map[eid] = []

    for i, (_, email) in enumerate(scored_users[:top_k]):
        rsvp = {
            "email": email,
            "status": "attended" if i % 5 == 0 else "joined"
        }
        if rsvp["status"] == "attended":
            rsvp["rating"] = random.randint(3, 5)
            rsvp["review"] = random.choice(sample_reviews)
        rsvp_map[eid].append(rsvp)

# Attach and save
for event in events:
    eid = event.get("id", event.get("title", ""))
    event["rsvpList"] = rsvp_map.get(eid, [])

    with open("events_with_rsvp_semantic.json", "w") as f:
        json.dump(events, f, indent=2)

print("✅ events_with_rsvp_semantic.json generated with location-aware RSVP matching.")