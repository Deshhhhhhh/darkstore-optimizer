import h3
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union

CITY_CENTER = (26.9124, 75.7873)   # Jaipur (lat, lng)
RES = 8
K = 15                              # rings of hexagons, about 12 km radius

# 1. Hexagons within K rings of the city center
center_cell = h3.latlng_to_cell(CITY_CENTER[0], CITY_CENTER[1], RES)
cells = list(h3.grid_disk(center_cell, K))
print(f"{len(cells)} hexagons created")

# 2. Convert hexagons to polygons
def cell_polygon(c):
    pts = h3.cell_to_boundary(c)          # (lat, lng) pairs
    return Polygon([(lng, lat) for lat, lng in pts])

grid = gpd.GeoDataFrame(
    {"h3": cells},
    geometry=[cell_polygon(c) for c in cells],
    crs="EPSG:4326",
)

cent = grid.to_crs(3857).geometry.centroid.to_crs(4326)
grid["lat"] = cent.y
grid["lng"] = cent.x

grid.to_file("data/processed/grid.geojson", driver="GeoJSON")

# 3. Save the study-area outline (Step 3 uses this)
outline = gpd.GeoDataFrame(geometry=[unary_union(grid.geometry)], crs="EPSG:4326")
outline.to_file("data/processed/boundary.geojson", driver="GeoJSON")
print("Saved grid.geojson and boundary.geojson")

# 4. Map you can open in your browser
m = grid.explore(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Esri, HERE, Garmin, OpenStreetMap contributors",
)
m.save("data/processed/grid_map.html")
print("Saved data/processed/grid_map.html")