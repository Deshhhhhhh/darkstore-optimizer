import pandas as pd
import geopandas as gpd
import folium

grid = gpd.read_file("data/processed/grid_scores.geojson").reset_index(drop=True)
cov = pd.read_csv("data/processed/coverage.csv")

best = int(grid["opportunity"].idxmax())
zone = cov[(cov["cand"] == best) & (cov["secs"] <= 600)]["hex"]
grid["in_zone"] = grid.index.isin(zone).astype(int)

tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
m = grid[["h3", "in_zone", "pop", "geometry"]].explore(
    column="in_zone", cmap="YlGn", vmin=0, vmax=1, tiles=tiles,
    attr="Esri, HERE, Garmin, OpenStreetMap contributors", legend=False,
    style_kwds=dict(weight=0.3, fillOpacity=0.6))

lat, lng = grid.loc[best, "lat"], grid.loc[best, "lng"]
folium.Marker([lat, lng], tooltip="Test store site").add_to(m)
m.save("data/processed/zone_test.html")

print(f"Test site: lat {lat:.4f}, lng {lng:.4f}")
print(f"Hexagons within 10 min: {int(grid['in_zone'].sum())}")
print(f"Population within 10 min: {int(grid.loc[grid['in_zone'] == 1, 'pop'].sum()):,}")
print("Saved zone_test.html")