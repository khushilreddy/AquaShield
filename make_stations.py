import pandas as pd, numpy as np

SRC = "data/processed/gwq_cleaned.csv"   # <- your cleaned file
d = pd.read_csv(SRC, low_memory=False)
d.columns = [str(c).strip().lower() for c in d.columns]

def find(*keys):
    for c in d.columns:
        if any(k in c for k in keys):
            return c

m = {
    "lat": find("latitude"), "lon": find("longitude"),
    "date": find("acquisition", "date"),
    "district": next((c for c in d.columns if "district" in c and "code" not in c), None),
    "station": find("station"), "iron": find("iron"),
    "tds": find("tds", "dissolved"), "chloride": find("chlor"),
    "sulphate": find("sulph", "so4"), "hardness": find("hardness"),
    "ec": find("conduct"),
    "ph": find("potential of hydrogen"),
}
print("column mapping:", m)

df = pd.DataFrame({k: d[v] for k, v in m.items() if v})
for c in ["lat", "lon", "iron", "tds", "chloride", "sulphate", "hardness", "ec", "ph"]:
    if c in df:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(r"[<>]", "", regex=True), errors="coerce")

if "date" in df:
    yr = pd.to_datetime(df["date"], errors="coerce").dt.year
    if yr.notna().any():
        df = df[yr >= 2005]
print("after year filter:", len(df))

if "district" in df:
    df = df[df["district"].astype(str).str.contains("barpeta|nalbari|kamrup", case=False, na=False)]
    print("after district filter:", len(df))
else:
    df["district"] = "unknown"

df = df.dropna(subset=["lat", "lon", "iron"])
df = df[df.lon.between(90.5, 92.2) & df.lat.between(25.8, 27.0)]
print("after box filter:", len(df))

agg = {c: "median" for c in ["lat", "lon", "tds", "chloride", "sulphate", "hardness", "ec", "ph"] if c in df}
agg["iron"] = "median"
agg["district"] = "first"
agg["date"] = "count"
out = df.groupby("station", as_index=False).agg(agg).rename(columns={"date": "n_samples"})
mx = df.groupby("station")["iron"].max().rename("iron_max")
out = out.merge(mx, on="station")
out["label_iron"] = (out["iron"] > 1.0).astype(int)
out.to_csv("data/processed/stations.csv", index=False)

print(len(out), "stations | positive rate:", round(out.label_iron.mean(), 3))
print(out.head(3).to_string())