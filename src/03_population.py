import numpy as np
import pandas as pd
import geopandas as gpd
import h3
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import xy

RES = 8
URL = ("https://worldpop-public-data.soton.ac.uk/GIS/Population/Global_2015_2030/R2025A/"
       "2024/IND/v1/100m/constrained/ind_pop_2024_CN_100m_R2025A_v1.tif")
grid = gpd.read_file("data/processed/grid_features.geojson")
minx, miny, maxx, maxy = grid.total_bounds
pad = 0.01

print("Reading Jaipur window from local WorldPop file...")
with rasterio.open("data/raw/ind_pop_2024.tif") as src:
    win = from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, src.transform)
    win = win.round_offsets().round_lengths()
    data = src.read(1, window=win)
    transform = src.window_transform(win)
    nodata = src.nodata

print("Window shape:", data.shape)

# keep valid, positive pixels only
mask = data > 0
if nodata is not None:
    mask &= data != nodata
rows, cols = np.where(mask)
xs, ys = xy(transform, rows, cols)
vals = data[rows, cols]

# assign each pixel to an H3 hexagon and sum population
cells = [h3.latlng_to_cell(y, x, RES) for x, y in zip(xs, ys)]
pop = pd.DataFrame({"h3": cells, "pop": vals}).groupby("h3")["pop"].sum()

grid["pop"] = grid["h3"].map(pop).fillna(0).round().astype(int)
print(f"Total population in study area: {grid['pop'].sum():,}")
print(f"Hexagons with zero population: {(grid['pop'] == 0).sum()}")

grid.to_file("data/processed/grid_pop.geojson", driver="GeoJSON")
grid.drop(columns="geometry").to_csv("data/processed/grid_pop.csv", index=False)
print("Saved grid_pop.geojson and grid_pop.csv")