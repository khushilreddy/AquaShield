import numpy as np, pandas as pd
from scipy.spatial import cKDTree

P = "data/processed/"
r = pd.read_csv(P + "rainfall_cleaned.csv", low_memory=False)
r.columns = [c.strip().lower() for c in r.columns]
r["mm"] = pd.to_numeric(r["telemetry hourly rainfall (mm)"], errors="coerce")
r = r[r["district"].astype(str).str.contains("barpeta|nalbari|kamrup", case=False, na=False)]
r = r.drop_duplicates(["station", "data acquisition time"])
r = r[(r.mm > 0) & (r.mm < 100)]

st = r.groupby(["station", "latitude", "longitude"], as_index=False).agg(
    rain_mm=("mm", "median"), mean_mm=("mm", "mean"), readings=("mm", "count"))
st = st[st.rain_mm < 20]          # drops NH17 Boko and NH27 Rangia (median ~50, sensor issue)
print(st.sort_values("rain_mm", ascending=False).round(2).to_string())
st.to_csv(P + "rain_stations.csv", index=False)

tree = cKDTree(st[["longitude", "latitude"]].values)
for name in ["features_wells.csv", "features_places.csv"]:
    try: d = pd.read_csv(P + name)
    except FileNotFoundError: print(name, "missing, skipped"); continue
    _, i = tree.query(d[["lon", "lat"]].values)
    d["rain_mm"] = st["rain_mm"].values[i]
    d.to_csv(P + name, index=False)
    print(name, "| min/max:", d.rain_mm.min().round(2), d.rain_mm.max().round(2), "| distinct:", d.rain_mm.nunique())