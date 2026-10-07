import os
import numpy as np
import pandas as pd
import geopandas as gpd
import h3
import folium

RES = 8
RINGS = 2   # catchment: hexagon + neighbours within 2 rings

W_DEMAND = {"pop": 0.45, "youth": 0.20, "activity": 0.10, "affluence": 0.25}
W_SUPPLY = {"grocery": 1.0, "qc": 3.0, "manual": 5.0}

grid = gpd.read_file("data/processed/grid_places.geojson")

# optional hand-collected competitor dark stores (columns: lat,lng)
grid["comp_manual"] = 0
if os.path.exists("data/raw/competitors.csv"):
    comp = pd.read_csv("data/raw/competitors.csv")
    cells = [h3.latlng_to_cell(r.lat, r.lng, RES) for r in comp.itertuples()]
    grid["comp_manual"] = grid["h3"].map(pd.Series(cells).value_counts()).fillna(0)
    print(f"Added {len(comp)} manual competitor locations")

def smooth(series):
    d = dict(zip(grid["h3"], series))
    return pd.Series(
        [sum(d.get(x, 0) for x in h3.grid_disk(h, RINGS)) for h in grid["h3"]],
        index=grid.index,
    )

def norm(s):
    cap = s.quantile(0.95)
    cap = cap if cap > 0 else s.max()
    return (s.clip(upper=cap) / cap) if cap > 0 else s * 0

# raw ingredients
grid["youth_raw"] = grid["colleges"] * 3 + grid["hostels"] * 3 + grid["offices"] + grid["malls"] * 2
grid["grocery_best"] = grid[["grocery", "grocery_ov"]].max(axis=1)   # avoid double counting
grid["supply_raw"] = (grid["grocery_best"] * W_SUPPLY["grocery"]
                      + grid["qc_stores"] * W_SUPPLY["qc"]
                      + grid["comp_manual"] * W_SUPPLY["manual"])

# catchment sums
pop_s = smooth(grid["pop"])
youth_s = smooth(grid["youth_raw"])
act_s = smooth(grid["poi_all"])
aff_s = smooth(grid["affluence_poi"])
sup_s = smooth(grid["supply_raw"])

grid["demand"] = (W_DEMAND["pop"] * norm(pop_s)
                  + W_DEMAND["youth"] * norm(youth_s)
                  + W_DEMAND["activity"] * norm(act_s)
                  + W_DEMAND["affluence"] * norm(aff_s)).round(3)
grid["supply"] = norm(sup_s).round(3)
grid["opportunity"] = (grid["demand"] * (1 - grid["supply"])).round(3)

print("\nTop 10 opportunity hexagons (check on Google Maps):")
top = grid.nlargest(10, "opportunity")[["lat", "lng", "pop", "demand", "supply", "opportunity"]]
print(top.to_string(index=False))

grid.to_file("data/processed/grid_scores.geojson", driver="GeoJSON")
grid.drop(columns="geometry").to_csv("data/processed/grid_scores.csv", index=False)

# map with switchable layers
tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
attr = "Esri, HERE, Garmin, OpenStreetMap contributors"
tip = ["pop", "demand", "supply", "opportunity"]
m = grid.explore(column="opportunity", cmap="YlOrRd", tiles=tiles, attr=attr,
                 name="Opportunity", tooltip=tip, legend=True,
                 style_kwds=dict(weight=0.3, fillOpacity=0.7))
grid.explore(column="demand", cmap="Blues", m=m, name="Demand", tooltip=tip,
             legend=False, show=False, style_kwds=dict(weight=0.3, fillOpacity=0.7))
grid.explore(column="supply", cmap="Greens", m=m, name="Supply", tooltip=tip,
             legend=False, show=False, style_kwds=dict(weight=0.3, fillOpacity=0.7))
folium.LayerControl().add_to(m)
m.save("data/processed/scores_map.html")
print("\nSaved grid_scores.csv and scores_map.html")