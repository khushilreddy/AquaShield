import json
from functools import lru_cache
from pathlib import Path
 
import folium
from folium.plugins import Fullscreen, HeatMap, MiniMap
 
DEFAULT_CENTER = (12.9, 77.6) 
 
CONTEXT_LAYERS = {
    "Flood extent": dict(
        path="data/raw/flood_extent.shp", show=True,
        style={"color": "#3a86ff", "fillColor": "#3a86ff", "fillOpacity": 0.25, "weight": 1}),
    "Waterways": dict(
        path="data/raw/waterways.shp", show=True,
        style={"color": "#0077b6", "weight": 2}),
    "Industrial sites": dict(
        path="data/raw/industrial_sites.shp", show=True,
        style={"color": "#6a040f", "fillColor": "#9d0208", "fillOpacity": 0.6, "weight": 2}),
    "Sewage outlets": dict(
        path="data/raw/sewage_outlets.shp", show=False,
        style={"color": "#5c4033", "fillColor": "#8d6e63", "fillOpacity": 0.7, "weight": 2}),
}
 
TYPE_ICONS = {"Village": "home", "Field": "leaf", "Well": "tint", "Facility": "plus-square"}
 
 
def risk_colour(score: float) -> str:
    """Hex colour for circles / legend."""
    if score >= 80:
        return "#d62828"
    if score >= 60:
        return "#f77f00"
    if score >= 40:
        return "#e6c200"
    return "#2a9d3a"
 
 
def risk_icon_colour(score: float) -> str:
    """folium.Icon only accepts named colours."""
    if score >= 80:
        return "red"
    if score >= 60:
        return "orange"
    if score >= 40:
        return "beige"
    return "green"
 
 
def popup_html(r) -> str:
    score = float(r["risk_score"])
    unc = float(r["uncertainty"])
    action = r.get("recommended_action", "n/a")
    return (
        f"<div style='font-family:sans-serif;font-size:13px;min-width:200px'>"
        f"<b>{r['name']}</b><br>"
        f"<span style='color:#666'>{r['type']} · {r['district']}</span>"
        f"<div style='background:#eee;border-radius:4px;margin:6px 0'>"
        f"<div style='width:{score:.0f}%;background:{risk_colour(score)};"
        f"color:#fff;padding:2px 4px;border-radius:4px'>{score:.0f}%</div></div>"
        f"Uncertainty: ±{unc:.0f}%<br>"
        f"<b>Action:</b> {action}</div>"
    )
 
 
@lru_cache(maxsize=None)
def _load_geojson(path: str):
    """Read a shapefile/GeoJSON once and cache it (Streamlit reruns often)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        import geopandas as gpd
    except ImportError:
        return None
    gdf = gpd.read_file(p)
    if gdf.crs is not None:
        gdf = gdf.to_crs(4326)
    # keep the file light for the browser
    if gdf.geom_type.isin(["Polygon", "MultiPolygon"]).any():
        gdf["geometry"] = gdf.geometry.simplify(0.0005)
    return json.loads(gdf.to_json())
 
 
def add_context_layers(m: folium.Map) -> None:
    for name, cfg in CONTEXT_LAYERS.items():
        data = _load_geojson(cfg["path"])
        if data is None:
            continue
        style = cfg["style"]
        folium.GeoJson(
            data, name=name, show=cfg["show"],
            style_function=lambda _f, s=style: s,
            marker=folium.CircleMarker(radius=5, **{
                k: v for k, v in style.items() if k in ("color", "fillColor", "fillOpacity")}),
            tooltip=name,
        ).add_to(m)
 
 
def add_legend(m: folium.Map) -> None:
    rows = "".join(
        f"<div><span style='background:{c};width:12px;height:12px;display:inline-block;"
        f"border-radius:50%;margin-right:6px'></span>{label}</div>"
        for c, label in [("#d62828", "High (≥ 80)"), ("#f77f00", "Medium-high (60-80)"),
                         ("#e6c200", "Medium (40-60)"), ("#2a9d3a", "Low (< 40)")]
    )
    html = (
    "<div style='position:fixed;bottom:25px;left:25px;z-index:9999;"
    "background:#fff;color:#000;opacity:1;"
    "padding:8px 10px;border-radius:6px;"
    "box-shadow:0 1px 5px rgba(0,0,0,.4);"
    "font:12px sans-serif;line-height:1.6;'>"
    "<b style='color:#000;'>Contamination risk</b>"
    + rows.replace("<div>", "<div style='color:#000;'>")
    + "</div>"
)
    m.get_root().html.add_child(folium.Element(html))
 
 
def build_map(df) -> folium.Map:
    empty = df is None or len(df) == 0
    centre = DEFAULT_CENTER if empty else (df["lat"].mean(), df["lon"].mean())
    m = folium.Map(location=centre, zoom_start=11, tiles="OpenStreetMap")
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite", show=False,
    ).add_to(m)
    Fullscreen().add_to(m)
    MiniMap(toggle_display=True, position="bottomright").add_to(m)
 
    add_context_layers(m)
 
    if not empty:
        heat = folium.FeatureGroup(name="Risk heatmap", show=True)
        HeatMap(
            [[r.lat, r.lon, r.risk_score / 100] for r in df.itertuples()],
            radius=28, blur=22, min_opacity=0.3,
            gradient={0.2: "#2a9d3a", 0.5: "#ffd60a", 0.8: "#d62828"},
        ).add_to(heat)
        heat.add_to(m)
 
        for t, sub in df.groupby("type"):
            group = folium.FeatureGroup(name=f"{t}s", show=True)
            icon_name = TYPE_ICONS.get(t, "info-sign")
            for _, r in sub.iterrows():
                folium.Marker(
                    [r["lat"], r["lon"]],
                    icon=folium.Icon(color=risk_icon_colour(r["risk_score"]),
                                     icon=icon_name, prefix="fa"),
                    tooltip=f"{r['name']} - {r['risk_score']:.0f}%",
                    popup=folium.Popup(popup_html(r), max_width=280),
                ).add_to(group)
            group.add_to(m)
 
        m.fit_bounds([[df["lat"].min(), df["lon"].min()],
                      [df["lat"].max(), df["lon"].max()]])
 
    folium.LayerControl(collapsed=False).add_to(m)
    add_legend(m)
    return m
 
 
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
 
    csv = Path("data/processed/risk_scores.csv")
    if csv.exists():
        data = pd.read_csv(csv)
    else:  
        rng = np.random.default_rng(1)
        n = 80
        data = pd.DataFrame({
            "id": range(n), "name": [f"Site {i}" for i in range(n)],
            "type": rng.choice(["Village", "Field", "Well", "Facility"], n),
            "district": "District A",
            "lat": 12.9 + rng.normal(0, .1, n), "lon": 77.6 + rng.normal(0, .1, n),
            "risk_score": rng.uniform(5, 98, n).round(0),
            "uncertainty": rng.uniform(8, 30, n).round(0),
            "recommended_action": "Water testing",
        })
    build_map(data).save("test_map.html")
    print("Saved test_map.html - open it in your browser.")