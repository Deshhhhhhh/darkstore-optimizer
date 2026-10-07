import geopandas as gpd
import pandas as pd
import h3

RES = 8
grid = gpd.read_file("data/processed/grid_pop.geojson")
places = gpd.read_parquet("data/raw/places.parquet")
print(f"Total places downloaded: {len(places):,}")
print("Columns:", list(places.columns))

def get(d, key):
    return d.get(key) if isinstance(d, dict) else None

# pick whichever category field this Overture release has
if "taxonomy" in places.columns:
    cat = places["taxonomy"].apply(lambda t: get(t, "primary"))
    print("Using category field: taxonomy.primary")
elif "basic_category" in places.columns:
    cat = places["basic_category"]
    print("Using category field: basic_category")
elif "categories" in places.columns:
    cat = places["categories"].apply(lambda c: get(c, "primary"))
    print("Using category field: categories.primary")
else:
    cat = pd.Series("unknown", index=places.index)
    print("WARNING: no category field found")

places["cat"] = cat.fillna("unknown").astype(str)
if "names" in places.columns:
    places["name"] = places["names"].apply(lambda n: get(n, "primary")).fillna("")
else:
    places["name"] = ""

print("\nTop 15 categories:")
print(places["cat"].value_counts().head(15).to_string())

# assign every place to a hexagon
places["h3"] = [h3.latlng_to_cell(g.y, g.x, RES) for g in places.geometry]

# groups we care about
grocery_words = "grocery|supermarket|convenience|fruit|vegetable|dairy"
is_grocery = places["cat"].str.contains(grocery_words, case=False, regex=True)
qc_names = "blinkit|zepto|instamart|bigbasket|bb now|dmart|d-mart|jiomart"
is_qc = places["name"].str.contains(qc_names, case=False, regex=True)

grid["poi_all"] = grid["h3"].map(places["h3"].value_counts()).fillna(0).astype(int)
grid["grocery_ov"] = grid["h3"].map(places[is_grocery]["h3"].value_counts()).fillna(0).astype(int)
grid["qc_stores"] = grid["h3"].map(places[is_qc]["h3"].value_counts()).fillna(0).astype(int)

affl_words = "cafe|coffee|gym|fitness|spa|salon|beauty|cosmetic|boutique|electronics"
is_affl = places["cat"].str.contains(affl_words, case=False, regex=True)
grid["affluence_poi"] = grid["h3"].map(places[is_affl]["h3"].value_counts()).fillna(0).astype(int)
print(f"Affluence-signal places: {grid['affluence_poi'].sum():,}")

print(f"\nGrocery-type places: {grid['grocery_ov'].sum():,}   (OSM found 70)")
print(f"Quick-commerce / big-chain stores: {grid['qc_stores'].sum():,}")
print(f"All places in grid: {grid['poi_all'].sum():,}")

print("\nTop 5 most populated hexagons (check these on Google Maps):")
print(grid.nlargest(5, "pop")[["lat", "lng", "pop"]].to_string(index=False))

grid.to_file("data/processed/grid_places.geojson", driver="GeoJSON")
grid.drop(columns="geometry").to_csv("data/processed/grid_places.csv", index=False)
print("\nSaved grid_places.geojson and grid_places.csv")