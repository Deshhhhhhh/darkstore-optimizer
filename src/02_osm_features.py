import osmnx as ox
import h3
import pandas as pd
import geopandas as gpd

RES = 8
ox.settings.timeout = 300

grid = gpd.read_file("data/processed/grid.geojson")
boundary = gpd.read_file("data/processed/boundary.geojson")
poly = boundary.geometry.iloc[0]

TAGS = {
    "colleges":   {"amenity": ["college", "university"]},
    "hostels":    {"tourism": "hostel", "building": "dormitory"},
    "offices":    {"office": True},
    "apartments": {"building": ["apartments", "residential"]},
    "malls":      {"shop": "mall"},
    "grocery":    {"shop": ["supermarket", "convenience", "grocery", "general"]},
}

def count_per_hex(tags):
    feats = ox.features_from_polygon(poly, tags)
    if feats.empty:
        return pd.Series(dtype=int)
    pts = feats.to_crs(3857).geometry.centroid.to_crs(4326)
    cells = [h3.latlng_to_cell(p.y, p.x, RES) for p in pts]
    return pd.Series(cells).value_counts()

for name, tags in TAGS.items():
    print(f"Fetching {name}...")
    try:
        counts = count_per_hex(tags)
    except Exception as e:
        print(f"  FAILED: {e}")
        counts = pd.Series(dtype=int)
    grid[name] = grid["h3"].map(counts).fillna(0).astype(int)
    print(f"  total found: {grid[name].sum()}")

grid.to_file("data/processed/grid_features.geojson", driver="GeoJSON")
grid.drop(columns="geometry").to_csv("data/processed/grid_features.csv", index=False)
print("Saved grid_features.geojson and grid_features.csv")