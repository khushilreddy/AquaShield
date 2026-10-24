import os
import numpy as np, pandas as pd, geopandas as gpd, rasterio

P = "data/processed/"
R = "data/raw/"

# ---------- slope from DEM ----------
if not os.path.exists(R + "slope.tif"):
    with rasterio.open(R + "dem.tif") as s:
        a = s.read(1).astype("float32")
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        lat = (s.bounds.top + s.bounds.bottom) / 2
        dx = abs(s.res[0]) * 111320 * np.cos(np.radians(lat))
        dy = abs(s.res[1]) * 110540
        gy, gx = np.gradient(a, dy, dx)
        sl = np.degrees(np.arctan(np.hypot(gx, gy))).astype("float32")
        prof = s.profile.copy(); prof.update(dtype="float32", nodata=np.nan)
    with rasterio.open(R + "slope.tif", "w", **prof) as d:
        d.write(sl, 1)
    print("slope.tif written")


def sample(path, lon, lat):
    with rasterio.open(path) as s:
        x, y = lon, lat
        if s.crs and s.crs.to_string() != "EPSG:4326":
            p = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs="EPSG:4326").to_crs(s.crs)
            x, y = p.x.values, p.y.values
        v = np.array([r[0] for r in s.sample(zip(x, y))], dtype="float64")
        if s.nodata is not None and not np.isnan(s.nodata):
            v[v == s.nodata] = np.nan
        v[v < -1000] = np.nan
        return v


def dist_km(pts_utm, layer_path):
    if not os.path.exists(layer_path):
        return np.full(len(pts_utm), np.nan)
    g = gpd.read_file(layer_path).to_crs("EPSG:32646")
    g = g[~g.geometry.is_empty & g.geometry.notna()][["geometry"]]
    if not len(g):
        return np.full(len(pts_utm), np.nan)
    j = gpd.sjoin_nearest(pts_utm[["geometry"]], g, distance_col="d")
    j = j[~j.index.duplicated()]
    return (j["d"].reindex(pts_utm.index) / 1000).values


def add_features(df):
    df = df.reset_index(drop=True)
    lon, lat = df["lon"].values, df["lat"].values
    df["elev"] = sample(R + "dem.tif", lon, lat)
    for name, f in [("twi", "twi.tif"), ("slope", "slope.tif"), ("flow_acc", "flow_acc.tif")]:
        if os.path.exists(R + f):
            df[name] = sample(R + f, lon, lat)
    pts = gpd.GeoDataFrame(df[["lon", "lat"]], geometry=gpd.points_from_xy(lon, lat),
                           crs="EPSG:4326").to_crs("EPSG:32646")
    df["d_waterway_km"] = dist_km(pts, R + "osm_waterways.gpkg")
    df["d_industry_km"] = dist_km(pts, R + "osm_industry.gpkg")
    df["d_wastewater_km"] = dist_km(pts, R + "osm_wastewater.gpkg")
    df["d_flood_km"] = dist_km(pts, P + "flood_extent.geojson")
    df["flooded"] = (df["d_flood_km"] == 0).astype(int)
    return df


def report(df, name):
    print(f"\n{name}: {len(df)} rows. NaN %:")
    print((df.isna().mean() * 100).round(1).to_string())


# ---------- wells ----------
w = pd.read_csv(P + "stations.csv")
w.columns = [c.strip().lower() for c in w.columns]
w = w.rename(columns={"latitude": "lat", "longitude": "lon"})
w = w.dropna(subset=["lat", "lon"])
wf = add_features(w)
wf.to_csv(P + "features_wells.csv", index=False)
report(wf, "features_wells")

# ---------- villages + facilities ----------
rows = []
for typ, f in [("village", "osm_villages.gpkg"), ("facility", "osm_facilities.gpkg")]:
    if not os.path.exists(R + f):
        print(f, "missing, skipped")
        continue
    g = gpd.read_file(R + f).to_crs("EPSG:32646")
    g["geometry"] = g.geometry.centroid
    g = g.to_crs("EPSG:4326")
    nm = g["name"] if "name" in g.columns else pd.Series([None] * len(g))
    rows.append(pd.DataFrame({"id": [f"{typ}_{i}" for i in range(len(g))],
                              "name": nm.fillna(typ).values, "type": typ,
                              "lon": g.geometry.x.values, "lat": g.geometry.y.values}))
if rows:
    pf = add_features(pd.concat(rows, ignore_index=True))
    pf.to_csv(P + "features_places.csv", index=False)
    report(pf, "features_places")