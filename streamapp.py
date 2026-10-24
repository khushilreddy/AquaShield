import html
import io
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import folium
    from streamlit_folium import st_folium
except ImportError:
    folium = None
    st_folium = None

try:
    from map_layers import build_map
except ImportError:
    try:
        from src.map_layers import build_map
    except ImportError:
        build_map = None

st.set_page_config(
    page_title="AquaShield",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path("data/processed/risk_scores.csv")
FACILITY_CSV_PATH = Path(__file__).with_name("emergency_facilities.csv")

TOP_FRACTION = 0.20

STATUSES = [
    "Pending collection",
    "Sample collected",
    "Sent to lab",
    "Results received",
    "Follow-up required",
]

FACTORS = {
    "flood_exposure": "Flood exposure",
    "dist_pollution_source": "Proximity to pollution source",
    "drainage_connectivity": "Drainage connectivity",
    "groundwater_vulnerability": "Groundwater vulnerability",
    "terrain_lowness": "Low-lying terrain",
}

ASSAM_DISTRICTS = [
    "BARPETA",
    "NALBARI",
    "KAMRUP",
    "KAMRUP METROPOLITAN",
]

FACILITY_COLUMNS = [
    "name",
    "type",
    "district",
    "address",
    "latitude",
    "longitude",
    "phone",
    "hours",
    "status",
    "capacity",
    "capacity_updated_at",
    "source",
    "last_verified",
    "flood_status",
    "notes",
]

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }

    .stApp {
        background: #07111f;
        color: #e5eefb;
    }

    [data-testid="stHeader"] {
        background: rgba(7, 17, 31, 0.96);
    }

    [data-testid="stSidebar"] {
        background: #0b1728;
        border-right: 1px solid #20354c;
    }

    [data-testid="stSidebar"] * {
        color: #e5eefb;
    }

    h1, h2, h3, h4 {
        color: #f0f7ff !important;
    }

    p, label, li, span {
        color: #c8d6e8;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #10263b, #0c1b2d);
        border: 1px solid #23445c;
        padding: 18px;
        border-radius: 15px;
    }

    [data-testid="stMetricLabel"] {
        color: #91abc6 !important;
    }

    [data-testid="stMetricValue"] {
        color: #5eead4 !important;
    }

    [data-testid="stMetricDelta"] {
        color: #7dd3fc !important;
    }

    div[data-baseweb="input"],
    div[data-baseweb="select"],
    div[data-baseweb="textarea"] {
        background: #102137;
        border-color: #29445f;
        border-radius: 8px;
    }

    div[data-baseweb="select"] * {
        color: #e5eefb;
    }

    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    .stTextArea textarea {
        background: #102137 !important;
        color: #f0f7ff !important;
        border: 1px solid #29445f !important;
    }

    .stSlider [data-baseweb="slider"] {
        color: #2dd4bf;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(135deg, #0d9488, #0369a1);
        color: #ffffff;
        border: 1px solid #2dd4bf;
        border-radius: 9px;
        font-weight: 600;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background: #0f766e;
        color: #ffffff;
        border-color: #5eead4;
    }

    div[data-testid="stTabs"] button {
        color: #b8cce2;
    }

    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #5eead4;
        border-bottom-color: #2dd4bf;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #203b55;
        border-radius: 10px;
    }

    .hero {
        background:
            radial-gradient(
                circle at top right,
                rgba(13, 148, 136, 0.20),
                transparent 35%
            ),
            linear-gradient(135deg, #10263b, #0b1728);
        border: 1px solid #24465e;
        border-radius: 20px;
        padding: 28px;
        margin-bottom: 22px;
    }

    .hero-title {
        font-size: 35px;
        font-weight: 800;
        color: #f0fdfa;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #a8bfd7;
        font-size: 15px;
    }

    .section-card {
        background: #0d1d30;
        border: 1px solid #233f58;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .priority-panel-title {
        font-size: 25px;
        line-height: 1.25;
        font-weight: 750;
        color: #f7f7f8;
        margin: 0 0 18px 0;
    }

    .priority-alert {
        background: #3b2529;
        border-radius: 9px;
        padding: 17px 18px;
        margin: 0 0 17px 0;
        color: #ff666b;
        font-size: 15px;
        line-height: 1.6;
        overflow-wrap: anywhere;
    }

    .priority-alert strong { color: #ff666b; }

    .priority-site {
        margin: 0 0 18px 0;
        padding: 0 0 1px 0;
        color: #f4f4f5;
        font-size: 15px;
        line-height: 1.65;
        overflow-wrap: anywhere;
    }

    .priority-site-heading {
        display: flex;
        flex-wrap: wrap;
        align-items: baseline;
        gap: 0 5px;
        margin-bottom: 2px;
        color: #f4f4f5;
    }

    .priority-site-dot { font-size: 17px; margin-right: 1px; }
    .priority-site-action { color: #f4f4f5; }
    .priority-site-action .arrow { font-weight: 700; }

    .risk-high {
        color: #fb7185;
        font-weight: 700;
    }

    .risk-medium {
        color: #fbbf24;
        font-weight: 700;
    }

    .risk-low {
        color: #5eead4;
        font-weight: 700;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    hr {
        border-color: #20354c;
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def make_demo_data(n=300, seed=7):

    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "id": [f"DEMO-{i + 1:04d}" for i in range(n)],
        "name": [f"Demo Location {i + 1}" for i in range(n)],
        "type": rng.choice(
            ["Village", "Field", "Well", "Facility"],
            size=n,
        ),
        "district": rng.choice(
            ["District A", "District B", "District C"],
            size=n,
        ),
        "lat": 12.9 + rng.normal(0, 0.15, n),
        "lon": 77.6 + rng.normal(0, 0.15, n),
        "flood_exposure": rng.uniform(0, 100, n),
        "dist_pollution_source": rng.uniform(0, 100, n),
        "drainage_connectivity": rng.uniform(0, 100, n),
        "groundwater_vulnerability": rng.uniform(0, 100, n),
        "terrain_lowness": rng.uniform(0, 100, n),
        "uncertainty": rng.uniform(5, 35, n),
    })

    factor_columns = list(FACTORS.keys())

    df["risk_score"] = (
        df[factor_columns].mean(axis=1) * 0.85
        + rng.uniform(0, 15, n)
    ).clip(0, 100)

    df["contaminated"] = (
        df["risk_score"] >= 65
    ).astype(int)

    df["main_source"] = (
        df[factor_columns]
        .idxmax(axis=1)
        .map(FACTORS)
    )

    return df

def load_data():

    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)

            if not df.empty:
                return df, False

        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            pass

    return make_demo_data(), True

def confidence_level(uncertainty):
    if pd.isna(uncertainty):
        return "Unknown"
    if uncertainty <= 15:
        return "High"
    if uncertainty <= 25:
        return "Medium"
    return "Low"

def priority_label(score):
    if pd.isna(score):
        return "Unranked"
    if score >= 80:
        return "P1"
    if score >= 60:
        return "P2"
    if score >= 40:
        return "P3"
    return "P4"

def recommend(row):
    score = row.get("risk_score", 0)
    confidence = row.get("confidence", "Unknown")

    if pd.isna(score):
        return "Validate data before assigning a sampling priority."

    if score >= 80:
        action = (
            "Prioritise field assessment and arrange water-quality "
            "testing as soon as practical."
        )
    elif score >= 60:
        action = (
            "Schedule water sampling and inspect nearby pollution "
            "and drainage conditions."
        )
    elif score >= 40:
        action = (
            "Monitor the location and include it in the next "
            "planned sampling round."
        )
    else:
        action = (
            "Continue routine monitoring and update the assessment "
            "when new observations are available."
        )

    if confidence == "Low":
        action += " Verify the input data because uncertainty is high."

    return action

def top_vs_random(df, frac=TOP_FRACTION, trials=500):

    if "contaminated" not in df.columns or df.empty:
        return None

    valid = df.dropna(subset=["contaminated", "risk_score"])

    if valid.empty:
        return None

    n_select = max(1, int(len(valid) * frac))
    top = valid.nlargest(n_select, "risk_score")
    top_rate = top["contaminated"].mean()

    rng = np.random.default_rng(7)
    random_rates = []

    for _ in range(trials):
        sample_indices = rng.choice(
            valid.index.to_numpy(),
            size=min(n_select, len(valid)),
            replace=False,
        )
        random_rates.append(
            valid.loc[sample_indices, "contaminated"].mean()
        )

    return {
        "top_rate": float(top_rate),
        "random_rate": float(np.mean(random_rates)),
        "n_select": n_select,
    }

def explain(row, available):

    values = {}

    for column, label in FACTORS.items():
        if column in available and pd.notna(row.get(column)):
            values[label] = float(row[column])

    factors = pd.Series(values, dtype=float).sort_values(
        ascending=False
    )

    if factors.empty:
        message = (
            "No factor-level data are available for this location."
        )
    else:
        strongest = factors.index[0]
        message = (
            f"The strongest recorded risk factor is {strongest}. "
            "Review this factor alongside field observations and "
            "water-quality results."
        )

    return message, factors

def load_facilities():

    if not FACILITY_CSV_PATH.exists():
        return pd.DataFrame(columns=FACILITY_COLUMNS), (
            "Emergency facility CSV was not found."
        )

    try:
        facilities = pd.read_csv(FACILITY_CSV_PATH)

    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame(columns=FACILITY_COLUMNS), (
            "The emergency facility CSV is empty or cannot be read."
        )

    if facilities.empty:
        return pd.DataFrame(columns=FACILITY_COLUMNS), (
            "The emergency facility CSV contains no records."
        )

    for column in FACILITY_COLUMNS:
        if column not in facilities.columns:
            facilities[column] = pd.NA

    facilities["district"] = (
        facilities["district"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    facilities["district"] = facilities["district"].replace({
        "KAMRUP METRO": "KAMRUP METROPOLITAN",
        "KAMRUP (METROPOLITAN)": "KAMRUP METROPOLITAN",
    })

    facilities = facilities[
        facilities["district"].isin(ASSAM_DISTRICTS)
    ].copy()

    facilities["latitude"] = pd.to_numeric(
        facilities["latitude"], errors="coerce"
    )
    facilities["longitude"] = pd.to_numeric(
        facilities["longitude"], errors="coerce"
    )
    facilities["capacity"] = pd.to_numeric(
        facilities["capacity"], errors="coerce"
    )

    valid_coordinates = (
        facilities["latitude"].between(-90, 90)
        & facilities["longitude"].between(-180, 180)
    )

    facilities.loc[
        ~valid_coordinates, ["latitude", "longitude"]
    ] = np.nan

    return facilities, None

def facility_map(facilities):

    if facilities.empty:
        st.info(
            "No facility records match the selected filters."
        )
        return

    mapped = facilities.dropna(
        subset=["latitude", "longitude"]
    ).copy()

    if mapped.empty:
        st.warning(
            "No matching facilities have valid coordinates. "
            "The table is available below, but the map cannot "
            "place these facilities accurately."
        )
        return

    if folium is None or st_folium is None:
        st.info(
            "Map display requires folium and streamlit-folium. "
            "Install them using the command shown below."
        )
        return

    center_lat = mapped["latitude"].mean()
    center_lon = mapped["longitude"].mean()

    map_object = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="CartoDB dark_matter",
    )

    for _, facility in mapped.iterrows():
        name = str(facility["name"])
        facility_type = str(facility["type"])
        district = str(facility["district"])

        popup_lines = [
            f"<b>{name}</b>",
            f"Type: {facility_type}",
            f"District: {district}",
        ]

        for column, label in [
            ("address", "Address"),
            ("phone", "Phone"),
            ("hours", "Hours"),
            ("status", "Status"),
            ("flood_status", "Flood status"),
        ]:
            value = facility.get(column)

            if pd.notna(value) and str(value).strip():
                popup_lines.append(
                    f"{label}: {str(value)}"
                )

        popup_html = "<br>".join(popup_lines)

        folium.Marker(
            location=[
                float(facility["latitude"]),
                float(facility["longitude"]),
            ],
            tooltip=name,
            popup=folium.Popup(
                popup_html,
                max_width=320,
            ),
        ).add_to(map_object)

    st_folium(
        map_object,
        height=480,
        use_container_width=True,
        returned_objects=[],
    )

REPORT_COLS = [
    "name",
    "type",
    "district",
    "risk_score",
    "uncertainty",
    "confidence",
    "priority",
    "recommended_action",
]

def report_html(sites, n_total, demo):
    generated = datetime.now().strftime("%d %b %Y, %H:%M")

    table = sites.reindex(columns=REPORT_COLS).to_html(
        index=False,
        escape=True,
        border=0,
    )

    demo_note = (
        "<p><strong>DEMO DATA:</strong> These records are "
        "synthetic and are not verified real-world assessments.</p>"
        if demo else ""
    )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>AquaShield Risk Report</title>
        <style>
            body {{
                background: #07111f;
                color: #e5eefb;
                font-family: Arial, sans-serif;
                padding: 30px;
            }}
            h1 {{ color: #5eead4; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                background: #102137;
            }}
            th, td {{
                padding: 10px;
                border: 1px solid #29445f;
                text-align: left;
            }}
            th {{ background: #0f766e; color: white; }}
            p {{ color: #c8d6e8; }}
        </style>
    </head>
    <body>
        <h1>AquaShield — Risk Report</h1>
        <p>Generated: {generated}</p>
        <p>Locations included: {len(sites)} of {n_total}</p>
        {demo_note}
        {table}
    </body>
    </html>
    """

def report_pdf(sites, n_total, demo):
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError:
        return None

    buffer = io.BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=24,
        leftMargin=24,
        topMargin=24,
        bottomMargin=24,
    )

    styles = getSampleStyleSheet()
    story = [
        Paragraph("AquaShield — Risk Report", styles["Title"]),
        Spacer(1, 10),
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}",
            styles["Normal"],
        ),
        Paragraph(
            f"Locations in report: {len(sites)} of {n_total}",
            styles["Normal"],
        ),
        Spacer(1, 12),
    ]

    if demo:
        story.append(
            Paragraph(
                "DEMO DATA: These records are synthetic and "
                "not verified real-world assessments.",
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 10))

    report_data = sites.reindex(columns=REPORT_COLS).fillna("")
    rows = [REPORT_COLS] + report_data.astype(str).values.tolist()

    table = Table(rows, repeatRows=1)

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9ca3af")),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
            colors.white,
            colors.HexColor("#e5e7eb"),
        ]),
    ]))

    story.append(table)
    document.build(story)

    buffer.seek(0)
    return buffer.getvalue()

df, is_demo = load_data()

for column in [
    "name",
    "type",
    "district",
    "risk_score",
    "uncertainty",
    *FACTORS.keys(),
]:
    if column not in df.columns:
        df[column] = np.nan

if "id" not in df.columns:
    df["id"] = [f"LOC-{i + 1:04d}" for i in range(len(df))]

df["risk_score"] = pd.to_numeric(
    df["risk_score"], errors="coerce"
)
df["uncertainty"] = pd.to_numeric(
    df["uncertainty"], errors="coerce"
)

df["confidence"] = df["uncertainty"].apply(confidence_level)
df["priority"] = df["risk_score"].apply(priority_label)
df["recommended_action"] = df.apply(recommend, axis=1)

facilities, facility_load_message = load_facilities()

with st.sidebar:
    st.markdown("## 💧 AquaShield")
    st.caption("Flood & groundwater risk intelligence")
    st.divider()

    st.markdown("### Dashboard filters")

    available_districts = sorted(
        df["district"].dropna().astype(str).unique()
    )

    selected_districts = st.multiselect(
        "District",
        options=available_districts,
        default=available_districts,
    )

    available_types = sorted(
        df["type"].dropna().astype(str).unique()
    )

    selected_types = st.multiselect(
        "Location type",
        options=available_types,
        default=available_types,
    )

    min_risk = st.slider(
        "Minimum risk score",
        min_value=0,
        max_value=100,
        value=0,
        step=5,
    )

    st.divider()
    st.caption(
        "Emergency help records are loaded separately from "
        "emergency_facilities.csv."
    )

view = df[
    df["district"].astype(str).isin(selected_districts)
    & df["type"].astype(str).isin(selected_types)
    & (df["risk_score"].fillna(-1) >= min_risk)
].copy()

view = view.sort_values(
    "risk_score",
    ascending=False,
    na_position="last",
)

priority_queue = view[
    view["priority"].isin(["P1", "P2"])
]

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">💧 AquaShield</div>
        <div class="hero-subtitle">
            Intelligent flood-risk monitoring, groundwater
            vulnerability assessment, and emergency facility discovery.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if is_demo:
    st.warning(
        "Demo mode is active. Risk locations and scores are synthetic, "
        "not real field observations. Add your actual risk_scores.csv "
        "to use real assessment data."
    )

tab_dash, tab_why, tab_track, tab_report, tab_emergency = st.tabs([
    "◈ Situation Room",
    "⌕ Risk Intelligence",
    "✓ Field Operations",
    "▤ Reports & Export",
    "🏥 Emergency Help",
])

with tab_dash:
    st.subheader("Situation Room")
    st.caption("Overview of the currently filtered risk dataset.")

    high_risk_count = int(
        (view["risk_score"] >= 80).sum()
    )

    positive_count = (
        int(view["contaminated"].sum())
        if "contaminated" in view.columns
        else None
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Locations in view",
        f"{len(view):,}",
    )

    metric2.metric(
        "High risk",
        f"{high_risk_count:,}",
    )

    metric3.metric(
        "Priority queue",
        f"{len(priority_queue):,}",
    )

    metric4.metric(
        "Positive labels",
        f"{positive_count:,}"
        if positive_count is not None
        else "N/A",
    )

    st.divider()

    left, right = st.columns([1.5, 1])

    with left:
        st.markdown("### Contamination risk map")

        map_data = view.dropna(
            subset=["lat", "lon"]
        ).copy() if {"lat", "lon"}.issubset(view.columns) else pd.DataFrame()

        if not map_data.empty and build_map is not None and st_folium is not None:
            try:
                risk_map = build_map(map_data)
                if folium is not None and hasattr(risk_map, "get_root"):
                    legend_css = folium.Element("""
                    <style>
                    .leaflet-control .legend,
                    .leaflet-control .info.legend,
                    .info.legend,
                    .legend {
                        background: rgba(255, 255, 255, 0.97) !important;
                        color: #111827 !important;
                        opacity: 1 !important;
                        border: 1px solid #9ca3af !important;
                        border-radius: 8px !important;
                        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.22) !important;
                        padding: 10px 12px !important;
                    }
                    .leaflet-control .legend *,
                    .leaflet-control .info.legend *,
                    .info.legend *,
                    .legend * {
                        color: #111827 !important;
                        opacity: 1 !important;
                        text-shadow: none !important;
                    }
                    .leaflet-control .legend i,
                    .leaflet-control .info.legend i,
                    .info.legend i,
                    .legend i {
                        opacity: 1 !important;
                    }
                    </style>
                    """)
                    risk_map.get_root().html.add_child(legend_css)
                st_folium(
                    risk_map,
                    height=545,
                    use_container_width=True,
                )
            except Exception as error:
                st.warning(f"The custom map could not be rendered: {error}")
                if {"lat", "lon"}.issubset(map_data.columns):
                    st.map(
                        map_data.rename(columns={"lat": "latitude", "lon": "longitude"}),
                        latitude="latitude",
                        longitude="longitude",
                    )
        elif not map_data.empty and {"lat", "lon"}.issubset(map_data.columns):
            st.map(
                map_data.rename(columns={"lat": "latitude", "lon": "longitude"}),
                latitude="latitude",
                longitude="longitude",
            )
        else:
            st.info("No usable coordinates are available for the filtered locations.")

    with right:
        st.markdown('<div class="priority-panel-title">Top priority sites &amp; actions</div>', unsafe_allow_html=True)

        if view.empty:
            st.info("No locations match the current filters.")
        else:
            top_row = view.iloc[0]
            top_score = top_row.get("risk_score")
            top_score_text = f"{top_score:.0f}" if pd.notna(top_score) else "N/A"
            top_name = html.escape(str(top_row.get("name", "Unnamed location")))
            top_district = html.escape(str(top_row.get("district", "Unknown district")))
            top_confidence = html.escape(str(top_row.get("confidence", "Unknown")))
            top_action = html.escape(str(top_row.get("recommended_action", "Review this location.")))

            if pd.notna(top_score) and top_score >= 80:
                alert_html = (f'<div class="priority-alert"><strong>Alert: {top_name} ({top_district}) - risk '
                              f'{top_score_text}/100, confidence {top_confidence}.</strong> {top_action}</div>')
                st.markdown(alert_html, unsafe_allow_html=True)

            for _, row in view.head(5).iterrows():
                score = row.get("risk_score")
                score_text = f"{score:.0f}" if pd.notna(score) else "N/A"
                name = html.escape(str(row.get("name", "Unnamed location")))
                confidence = html.escape(str(row.get("confidence", "Unknown")))
                action = html.escape(str(row.get("recommended_action", "Review this location.")))

                if pd.notna(score) and score >= 80:
                    dot = "🔴"
                elif pd.notna(score) and score >= 60:
                    dot = "🟠"
                else:
                    dot = "🟡"

                item_html = (f'<div class="priority-site"><div class="priority-site-heading">'
                             f'<span class="priority-site-dot">{dot}</span><strong>{name}</strong>'
                             f'<span>· {score_text}/100 · confidence <strong>{confidence}</strong></span></div>'
                             f'<div class="priority-site-action">→ {action}</div></div>')
                st.markdown(item_html, unsafe_allow_html=True)

            st.caption("Recommendations are advisory and require field verification.")

    st.divider()

    st.markdown("### Ranked sampling queue")

    queue_columns = [
        "name",
        "type",
        "district",
        "risk_score",
        "confidence",
        "priority",
        "recommended_action",
    ]

    st.dataframe(
        view.reindex(columns=queue_columns),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇ Download filtered risk data (CSV)",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="aquashield_risk_data.csv",
        mime="text/csv",
    )

    st.divider()

    st.markdown("### Risk prioritisation check")

    comparison = top_vs_random(view)

    if comparison is None:
        st.info(
            "A comparison requires a contaminated column with "
            "valid labels."
        )
    else:
        comparison_df = pd.DataFrame({
            "Selection": [
                "Highest-risk locations",
                "Random baseline",
            ],
            "Positive rate (%)": [
                comparison["top_rate"] * 100,
                comparison["random_rate"] * 100,
            ],
        }).set_index("Selection")

        st.bar_chart(comparison_df)

        st.caption(
            "This comparison uses the supplied labels. It is an "
            "exploratory evaluation, not proof that the model will "
            "perform similarly in the field."
        )

with tab_why:
    st.subheader("Risk Intelligence")
    st.write(
        "Inspect the factors contributing to a location's recorded "
        "risk score."
    )

    if view.empty:
        st.info("No locations match the current filters.")
    else:
        selected_name = st.selectbox(
            "Select location",
            options=view["name"].fillna("Unnamed location").tolist(),
        )

        selected_row = view[
            view["name"].fillna("Unnamed location") == selected_name
        ].iloc[0]

        detail1, detail2, detail3 = st.columns(3)

        score = selected_row.get("risk_score")
        uncertainty = selected_row.get("uncertainty")

        detail1.metric(
            "Risk score",
            f"{score:.1f}" if pd.notna(score) else "N/A",
        )

        detail2.metric(
            "Confidence",
            selected_row.get("confidence", "Unknown"),
        )

        detail3.metric(
            "Priority",
            selected_row.get("priority", "Unranked"),
        )

        st.markdown("### Explanation")

        explanation, factor_values = explain(
            selected_row,
            view.columns,
        )

        st.info(explanation)

        if not factor_values.empty:
            st.markdown("### Risk factor values")
            st.bar_chart(factor_values)

        st.markdown("### Recommended next step")
        st.write(selected_row["recommended_action"])

        st.caption(
            "Recommendations are rule-based guidance. Confirm "
            "important decisions using local observations and "
            "appropriate water-quality testing."
        )

with tab_track:
    st.subheader("Field Operations")
    st.write(
        "Track sampling and laboratory progress for locations "
        "in the current view."
    )

    if "field_status" not in st.session_state:
        st.session_state.field_status = {}

    if view.empty:
        st.info("No locations match the current filters.")
    else:
        tracker = view[
            ["id", "name", "district", "risk_score", "priority"]
        ].copy()

        tracker["status"] = tracker["id"].map(
            st.session_state.field_status
        ).fillna("Pending collection")

        tracker["status"] = tracker["status"].where(
            tracker["status"].isin(STATUSES),
            "Pending collection",
        )

        edited_tracker = st.data_editor(
            tracker,
            use_container_width=True,
            hide_index=True,
            disabled=[
                "id",
                "name",
                "district",
                "risk_score",
                "priority",
            ],
            column_config={
                "status": st.column_config.SelectboxColumn(
                    "Sampling status",
                    options=STATUSES,
                    required=True,
                )
            },
            key="field_tracker_editor",
        )

        if st.button("Save field statuses"):
            for _, row in edited_tracker.iterrows():
                st.session_state.field_status[row["id"]] = row["status"]

            st.success("Field statuses saved for this session.")

        st.download_button(
            "⬇ Export field tracker",
            data=edited_tracker.to_csv(index=False).encode("utf-8"),
            file_name="aquashield_field_tracker.csv",
            mime="text/csv",
        )

        st.caption(
            "Statuses are held in this Streamlit session. Export the "
            "CSV if you need to retain a copy outside the session."
        )

with tab_report:
    st.subheader("Reports & Export")

    report_count = st.slider(
        "Number of top-risk locations",
        min_value=1,
        max_value=max(1, len(view)),
        value=min(20, max(1, len(view))),
    )

    report_sites = view.head(report_count).copy()

    st.dataframe(
        report_sites.reindex(columns=REPORT_COLS),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "⬇ Download HTML report",
        data=report_html(
            report_sites,
            len(view),
            is_demo,
        ).encode("utf-8"),
        file_name="aquashield_report.html",
        mime="text/html",
    )

    pdf_data = report_pdf(
        report_sites,
        len(view),
        is_demo,
    )

    if pdf_data is not None:
        st.download_button(
            "⬇ Download PDF report",
            data=pdf_data,
            file_name="aquashield_report.pdf",
            mime="application/pdf",
        )
    else:
        st.info(
            "To enable PDF reports, install reportlab in your "
            "project environment."
        )

with tab_emergency:
    st.subheader("🏥 Emergency Help Locator")

    st.write(
        "Find emergency facilities from the supplied facility dataset "
        "for Barpeta, Nalbari, Kamrup, and Kamrup Metropolitan, Assam."
    )

    st.warning(
        "Use verified facility information for real emergencies. "
        "This dashboard does not confirm live availability, road access, "
        "or current facility capacity."
    )

    if facility_load_message:
        st.info(facility_load_message)

    if facilities.empty:
        st.info(
            "No facility records are available for the four supported "
            "districts. Add verified records to emergency_facilities.csv "
            "to populate this section."
        )

    else:
        filter1, filter2, filter3 = st.columns(3)

        with filter1:
            facility_districts = st.multiselect(
                "District",
                options=ASSAM_DISTRICTS,
                default=ASSAM_DISTRICTS,
                key="facility_district_filter",
            )

        with filter2:
            available_facility_types = sorted(
                facilities["type"]
                .dropna()
                .astype(str)
                .unique()
            )

            facility_types = st.multiselect(
                "Facility type",
                options=available_facility_types,
                default=available_facility_types,
                key="facility_type_filter",
            )

        with filter3:
            available_statuses = sorted(
                facilities["status"]
                .dropna()
                .astype(str)
                .unique()
            )

            facility_statuses = st.multiselect(
                "Facility status",
                options=available_statuses,
                default=available_statuses,
                key="facility_status_filter",
            )

        filtered_facilities = facilities[
            facilities["district"].isin(facility_districts)
        ].copy()

        if facility_types:
            filtered_facilities = filtered_facilities[
                filtered_facilities["type"]
                .astype(str)
                .isin(facility_types)
            ]
        else:
            filtered_facilities = filtered_facilities.iloc[0:0]

        if facility_statuses:
            filtered_facilities = filtered_facilities[
                filtered_facilities["status"]
                .astype(str)
                .isin(facility_statuses)
            ]
        else:
            filtered_facilities = filtered_facilities.iloc[0:0]

        valid_coords = filtered_facilities.dropna(
            subset=["latitude", "longitude"]
        )

        active_mask = (
            filtered_facilities["status"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["active", "open", "operational"])
        )

        unknown_capacity = int(
            filtered_facilities["capacity"].isna().sum()
        )

        m1, m2, m3, m4 = st.columns(4)

        m1.metric(
            "Facilities listed",
            len(filtered_facilities),
        )

        m2.metric(
            "Marked operational",
            int(active_mask.sum()),
        )

        m3.metric(
            "With coordinates",
            len(valid_coords),
        )

        m4.metric(
            "Unknown capacity",
            unknown_capacity,
        )

        st.divider()

        st.markdown("### Facility map")

        facility_map(filtered_facilities)

        st.markdown("### Facility directory")

        display_columns = [
            "name",
            "type",
            "district",
            "address",
            "phone",
            "hours",
            "status",
            "capacity",
            "capacity_updated_at",
            "flood_status",
            "last_verified",
            "source",
            "notes",
        ]

        st.dataframe(
            filtered_facilities.reindex(
                columns=display_columns
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "⬇ Download filtered facility list",
            data=filtered_facilities.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="aquashield_emergency_facilities.csv",
            mime="text/csv",
        )

    st.divider()

    st.markdown("### Facility data template")

    st.write(
        "Download the blank CSV template, fill it with verified "
        "facility details, and save it as emergency_facilities.csv "
        "in the same folder as streamapp.py."
    )

    template_df = pd.DataFrame(columns=FACILITY_COLUMNS)

    st.download_button(
        "⬇ Download blank facility CSV template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="emergency_facilities_template.csv",
        mime="text/csv",
    )

    with st.expander("Required CSV columns"):
        st.code(",".join(FACILITY_COLUMNS), language="text")

        st.markdown(
            """
            - **name, type, district:** facility details.
            - **address, latitude, longitude:** location details.
            - **phone, hours, status:** contact and operating information.
            - **capacity, capacity_updated_at:** capacity information and
              when it was last updated.
            - **source, last_verified:** provenance and verification date.
            - **flood_status, notes:** additional operational information.

            Only enter real details from reliable sources. Leave unknown
            fields blank rather than guessing. Keep Kamrup and Kamrup
            Metropolitan as separate districts.
            """
        )

st.divider()

st.caption(
    "AquaShield | Flood and groundwater risk intelligence | "
    "Verify field data before making operational decisions."
)