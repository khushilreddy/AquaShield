import os
import time

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import rasterio

UA = "AquaShield-hackathon/1.0 (student project; your_email@example.com)"

if hasattr(ox.settings, "requests_headers"):      # osmnx 2.x
    ox.settings.requests_headers = {"User-Agent": UA, "referer": "https://github.com/"}
else:                                             # osmnx 1.x
    ox.settings.default_user_agent = UA
    ox.settings.default_referer = "https://github.com/"

ox.settings.timeout = 300
ox.settings.use_cache = True
ox.settings.log_console = False

os.makedirs("data/raw", exist_ok=True)

with rasterio.open("data/raw/dem.tif") as s:
    W, S, E, N = s.bounds
print("Area:", round(W, 2), round(S, 2), round(E, 2), round(N, 2))

WATER = ["river", "stream", "canal", "drain"]
WASTE = ["wastewater_plant", "works"]
PLACES = ["village", "hamlet", "town", "suburb"]
FAC = ["hospital", "clinic", "school"]

# ONE combined query per tile (farmland skipped to save time)
TAGS = {"waterway": WATER, "landuse": ["industrial"], "man_made": WASTE,
        "place": PLACES, "amenity": FAC}
TAGCOLS = list(TAGS.keys())

STEP = 0.5
xs = np.arange(W, E, STEP)
ys = np.arange(S, N, STEP)
print("tiles:", len(xs) * len(ys))


def fetch(w, s, e, n):
    for attempt in range(3):
        try:
            try:
                return ox.features_from_bbox(bbox=(w, s, e, n), tags=TAGS)   # osmnx 2.x
            except TypeError:
                return ox.features_from_bbox(n, s, e, w, TAGS)               # osmnx 1.x
        except Exception as ex:
            if "InsufficientResponse" in type(ex).__name__ or "No matching features" in str(ex):
                return gpd.GeoDataFrame()
            print(f"    retry {attempt+1}: {str(ex)[:60]}")
            time.sleep(8)
    return gpd.GeoDataFrame()


parts = []
i = 0
for x in xs:
    for y in ys:
        i += 1
        w, e = x, min(x + STEP, E)
        s, n = y, min(y + STEP, N)
        g = fetch(w, s, e, n)
        print(f"tile {i}/{len(xs)*len(ys)}: {len(g)} features", flush=True)
        if len(g):
            g = g.reset_index(drop=True)
            cols = [c for c in ["name"] + TAGCOLS + ["geometry"] if c in g.columns]
            parts.append(g[cols])

allg = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs="EPSG:4326")
for c in TAGCOLS:
    if c not in allg.columns:
        allg[c] = None
allg["_k"] = allg.geometry.to_wkb()
allg = allg.drop_duplicates("_k").drop(columns="_k")

out = {
    "waterways":  allg[allg["waterway"].isin(WATER)],
    "industry":   allg[allg["landuse"] == "industrial"],
    "wastewater": allg[allg["man_made"].isin(WASTE)],
    "villages":   allg[allg["place"].isin(PLACES)],
    "facilities": allg[allg["amenity"].isin(FAC)],
}
for name, g in out.items():
    keep = [c for c in ["name", "geometry"] if c in g.columns]
    if len(g):
        g[keep].to_file(f"data/raw/osm_{name}.gpkg", driver="GPKG")
    print(name, "->", len(g), "features saved")