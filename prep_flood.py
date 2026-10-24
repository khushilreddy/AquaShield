import rasterio, numpy as np, geopandas as gpd, pandas as pd
from rasterio.features import shapes
from shapely.geometry import shape

SCALE = 4
parts = []
for f in ["data/raw/flood_A2_20220622.tif", "data/raw/flood_S1_20220621.tif"]:
    with rasterio.open(f) as s:
        a = s.read(1, out_shape=(s.height // SCALE, s.width // SCALE))
        tr = s.transform * s.transform.scale(SCALE, SCALE)
        vals = np.unique(a).tolist()
        print(f, "| crs:", s.crs, "| nodata:", s.nodata, "| values:", vals[:10])
        if not set(vals) <= {0, 1, s.nodata}:
            raise SystemExit("Values are not 0/1. Paste this line to me before continuing.")
        m = a == 1
        print("  flooded share:", round(m.mean() * 100, 2), "%")
        geoms = [shape(g) for g, v in shapes(m.astype("uint8"), mask=m, transform=tr)]
        parts.append(gpd.GeoDataFrame(geometry=geoms, crs=s.crs).to_crs("EPSG:4326"))

g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
u = g.geometry.union_all() if hasattr(g.geometry, "union_all") else g.geometry.unary_union
out = gpd.GeoDataFrame(geometry=[u], crs="EPSG:4326")
out["geometry"] = out.geometry.simplify(0.0005)
out.to_file("data/processed/flood_extent.geojson", driver="GeoJSON")
print("saved flood_extent.geojson")