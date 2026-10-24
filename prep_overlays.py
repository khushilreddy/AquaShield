import os
import geopandas as gpd

D = "data/raw/"
O = "data/processed/overlays/"
os.makedirs(O, exist_ok=True)

for name in ["waterways", "industry", "wastewater", "facilities", "villages"]:
    g = gpd.read_file(f"{D}osm_{name}.gpkg")
    g = g[[c for c in ["osm_id", "name", "fclass", "geometry"] if c in g.columns]]
    g["geometry"] = g.geometry.simplify(0.0005)
    g.to_file(f"{O}{name}.geojson", driver="GeoJSON")
    print(name, len(g))
