import os
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox

SPEED_KMH = 20          # average rider speed: an assumption, state it in the README
MAX_MIN = 15            # compute up to 15 minutes; filter lower in the dashboard
MAX_SECS = MAX_MIN * 60
GRAPH_FILE = "data/raw/jaipur_drive.graphml"

ox.settings.timeout = 300
grid = gpd.read_file("data/processed/grid_scores.geojson").reset_index(drop=True)

# 1. Road network (downloaded once, then cached)
if os.path.exists(GRAPH_FILE):
    print("Loading cached road network...")
    G = ox.load_graphml(GRAPH_FILE)
else:
    print("Downloading road network (a few minutes)...")
    boundary = gpd.read_file("data/processed/boundary.geojson")
    poly = boundary.geometry.iloc[0].buffer(0.03)   # include roads just outside the edge
    G = ox.graph_from_polygon(poly, network_type="drive")
    G = ox.truncate.largest_component(G, strongly=True)
    ox.save_graphml(G, GRAPH_FILE)
print(f"Road network: {len(G.nodes):,} nodes, {len(G.edges):,} edges")

# 2. Travel time per road segment (seconds) at a constant rider speed
mps = SPEED_KMH * 1000 / 3600
for u, v, k, d in G.edges(keys=True, data=True):
    d["tt"] = float(d["length"]) / mps

# 3. Snap every hexagon centre to its nearest road node
nodes = ox.distance.nearest_nodes(G, X=grid["lng"].values, Y=grid["lat"].values)
node_to_hexes = {}
for i, n in enumerate(nodes):
    node_to_hexes.setdefault(n, []).append(i)

# 4. From every candidate site, find the travel time to every reachable hexagon
rows = []
n_total = len(grid)
for i, src in enumerate(nodes):
    lengths = nx.single_source_dijkstra_path_length(G, src, cutoff=MAX_SECS, weight="tt")
    for node, secs in lengths.items():
        for j in node_to_hexes.get(node, []):
            rows.append((i, j, round(secs)))
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{n_total} candidate sites done")

cov = pd.DataFrame(rows, columns=["cand", "hex", "secs"])
cov = cov.groupby(["cand", "hex"], as_index=False)["secs"].min()
cov.to_csv("data/processed/coverage.csv", index=False)
print(f"\nSaved coverage.csv with {len(cov):,} rows")

# 5. Sanity check: how many hexagons does one site reach at 6, 8, 10, 12, 15 minutes?
for m in [6, 8, 10, 12, 15]:
    per_site = cov[cov["secs"] <= m * 60].groupby("cand")["hex"].nunique()
    print(f"  {m:>2} min: avg {per_site.mean():.0f} hexagons reached "
          f"(min {per_site.min()}, max {per_site.max()})")

# 6. Map of the 10-minute zone from the top-opportunity hexagon
best = int(grid["opportunity"].idxmax())
zone = cov[(cov["cand"] == best) & (cov["secs"] <= 600)]["hex"]
grid["in_zone"] = grid.index.isin(zone).astype(int)
grid["is_site"] = (grid.index == best).astype(int)
tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
m = grid[["h3", "in_zone", "is_site", "pop", "geometry"]].explore(
    column="in_zone", cmap=["#dddddd", "#2a9d8f"], tiles=tiles,
    attr="Esri, HERE, Garmin, OpenStreetMap contributors", legend=False,
    style_kwds=dict(weight=0.3, fillOpacity=0.6))
m.save("data/processed/zone_test.html")
print(f"\nTest site: lat {grid.loc[best,'lat']:.4f}, lng {grid.loc[best,'lng']:.4f}")
print(f"Population within 10 min of test site: {int(grid.loc[grid['in_zone']==1,'pop'].sum()):,}")
print("Saved zone_test.html")