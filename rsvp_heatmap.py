import json
import folium
from folium.plugins import MarkerCluster

# Load RSVP event data
with open("events_with_rsvp_semantic.json") as f:
    events = json.load(f)

# Create map centered around Phoenix metro (adjust as needed)
rsvp_map = folium.Map(location=[33.4255, -111.9400], zoom_start=10)
marker_cluster = MarkerCluster().add_to(rsvp_map)

# Plot each event with RSVP count and avg rating
for event in events:
    loc = event.get("location", {})
    lat = loc.get("latitude")
    lon = loc.get("longitude")
    if lat is None or lon is None:
        continue

    rsvps = event.get("rsvpList", [])
    ratings = [r.get("rating") for r in rsvps if r.get("status") == "attended" and "rating" in r]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else "N/A"

    popup_text = f"""
    <strong>{event.get('title')}</strong><br>
    City: {loc.get('city')}<br>
    RSVPs: {len(rsvps)}<br>
    Avg Rating: {avg_rating}
    """

    folium.CircleMarker(
        location=[lat, lon],
        radius=min(len(rsvps) / 5, 15),  # size based on RSVP count
        popup=popup_text,
        color="purple",
        fill=True,
        fill_opacity=0.7
    ).add_to(marker_cluster)

# Save map to file
rsvp_map.save("rsvp_heatmap.html")
print("✅ Map saved as rsvp_heatmap.html")