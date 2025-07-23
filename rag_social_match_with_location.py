import json
import faiss
from sentence_transformers import SentenceTransformer
from geopy.distance import geodesic
from geopy.geocoders import Nominatim

# Load data
with open("dummy_users.json") as f:
    users = json.load(f)
with open("events_with_rsvp_semantic.json") as f:
    events = json.load(f)

# Map users to events they attended
user_events = {}
for event in events:
    eid = event.get("id", event.get("title", ""))
    for rsvp in event.get("rsvpList", []):
        if rsvp.get("status") == "attended":
            user_events.setdefault(rsvp["email"], []).append(event.get("title"))

# Prepare embedding input for each user
embedder = SentenceTransformer("all-MiniLM-L6-v2")
user_chunks = []
user_emails = []
user_metadata = []

for user in users:
    email = user["email"]
    bio = user.get("bio", "")
    interests = ", ".join(user.get("interests", []))
    loc = user.get("location", {})
    lat = loc.get("latitude")
    lon = loc.get("longitude")
    city = loc.get("city", "")
    state = loc.get("state", "")
    attended = user_events.get(email, [])
    attended_str = ", ".join([title for title in attended[:5] if title]) if attended else "None"

    text = f"{bio}. Interests: {interests}. Location: {city}, {state}. Attended events: {attended_str}."
    user_chunks.append(text)
    user_emails.append(email)
    user_metadata.append({
        "email": email,
        "city": city,
        "state": state,
        "latitude": lat,
        "longitude": lon,
        "attended": attended
    })

# Embed and index
embeddings = embedder.encode(user_chunks, show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Geolocate helper
def get_coordinates_for_city(city_name):
    try:
        geolocator = Nominatim(user_agent="sahana-rag")
        location = geolocator.geocode(city_name)
        if location:
            return (location.latitude, location.longitude)
    except:
        return None
    return None

# Run search
print("🔍 Ask a question like: 'I like rock music and hiking' (type 'exit' to quit)")

while True:
    query = input("\n🧠 Your query: ").strip()
    if query.lower() in ["exit", "quit"]:
        break

    city_input = input("📍 Enter your city (for distance filtering): ").strip()
    query_coords = get_coordinates_for_city(city_input)

    if not query_coords:
        print("❌ Could not resolve location. Showing all results.")
        query_coords = None

    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, 50)

    print("\n👥 Top Matching Users:")
    for result_idx, idx in enumerate(I[0]):
        meta = user_metadata[idx]
        user_coords = (meta["latitude"], meta["longitude"])
        score = round(1 / (1 + D[0][result_idx]), 4)

        if query_coords and all(user_coords):
            dist = geodesic(query_coords, user_coords).km
            if dist > 50:
                continue
        else:
            dist = "N/A"

        print(f"- {meta['email']} ({meta['city']}) | Score: {score}, Distance: {round(dist, 1) if isinstance(dist, float) else 'N/A'} km")
        print(f"  Attended: {', '.join([title for title in meta['attended'] if title]) if meta['attended'] else 'None'}\n")