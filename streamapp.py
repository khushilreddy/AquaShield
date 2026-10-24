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

# Results file from the modelling step. Looked up in the current folder, next
# to this script, and in data/processed/. The newest file name comes first.
CSV_CANDIDATES = [
    "risk_scores_priority_1.csv",
    "risk_scores_priority.csv",
    "data/processed/risk_scores_priority_1.csv",
    "data/processed/risk_scores_priority.csv",
    "data/processed/risk_scores.csv",
]
REQUIRED_COLUMNS = [
    "id", "name", "type", "lat", "lon",
    "risk", "uncertainty", "rank", "tier",
]
# Columns that may or may not be present; created empty when missing.
OPTIONAL_COLUMNS = [
    "drivers", "flood_status", "district", "station", "n_samples",
    "label_iron", "chem_score", "chem_exceedances",
    "d_flood_km", "d_industry_km", "d_wastewater_km", "d_waterway_km",
    "rain_mm", "elev", "twi", "slope",
    "tds", "chloride", "sulphate", "hardness", "ec", "ph", "iron", "iron_max",
]
FACILITY_CSV_PATH = Path(__file__).with_name("emergency_facilities.csv")

TOP_FRACTION = 0.20

STATUSES = [
    "Pending collection",
    "Sample collected",
    "Sent to lab",
    "Results received",
    "Follow-up required",
]

# tier (from the model) -> short priority label used across the dashboard
TIER_TO_PRIORITY = {
    "Priority testing": "P1",
    "Routine testing": "P2",
}
TIER_ORDER = list(TIER_TO_PRIORITY.keys())

TIER_ACTIONS = {
    "Priority testing": (
        "Prioritise field assessment and arrange water-quality "
        "testing in the next sampling round."
    ),
    "Routine testing": (
        "Include in routine monitoring and update the assessment "
        "when new observations or lab results are available."
    ),
}

FACTOR_LABELS = {
    "d_flood_km": "Distance to June 2022 flood extent (km)",
    "d_industry_km": "Distance to nearest mapped industry (km)",
    "d_wastewater_km": "Distance to nearest wastewater site (km)",
    "d_waterway_km": "Distance to nearest river/drain (km)",
    "rain_mm": "Long-term rainfall index",
    "elev": "Elevation",
    "slope": "Slope",
    "twi": "Topographic wetness index (TWI)",
}

# Lab measurements available for the monitored wells (units as in the source data)
LAB_LABELS = {
    "tds": "TDS",
    "chloride": "Chloride",
    "sulphate": "Sulphate",
    "hardness": "Hardness",
    "ec": "Electrical conductivity (EC)",
    "ph": "pH",
    "iron": "Iron (mean)",
    "iron_max": "Iron (highest reading)",
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


# ---------------------------------------------------------------- data ----
def make_demo_data(n=300, seed=7):
    """Synthetic placeholder in the SAME format as the real results CSV."""
    rng = np.random.default_rng(seed)

    risk = np.clip(rng.beta(2, 3, n), 0.01, 0.99)
    uncertainty = rng.uniform(0.07, 0.43, n)
    priority = risk * (1 - 0.5 * uncertainty)
    rank = pd.Series(priority).rank(ascending=False, method="first").astype(int)

    return pd.DataFrame({
        "id": [f"DEMO-{i + 1:04d}" for i in range(n)],
        "name": [f"Demo Location {i + 1}" for i in range(n)],
        "type": rng.choice(["village", "well", "facility"], size=n),
        "lat": 26.4 + rng.normal(0, 0.2, n),
        "lon": 91.0 + rng.normal(0, 0.3, n),
        "risk": risk.round(4),
        "uncertainty": uncertainty.round(4),
        "rank": rank.values,
        "tier": np.where(rank <= n * 0.15, "Priority testing", "Routine testing"),
        "drivers": rng.choice([
            "near June 2022 flood extent; next to a river/drain",
            "near mapped industry; high long-term rainfall area",
            "near mapped wastewater site; next to a river/drain",
            "high model risk; high long-term rainfall area",
        ], size=n),
        "flood_status": rng.choice(["Watch", "Normal"], size=n, p=[0.15, 0.85]),
        "d_flood_km": rng.uniform(0, 80, n),
        "d_industry_km": rng.uniform(0, 45, n),
        "d_wastewater_km": rng.uniform(0, 130, n),
        "d_waterway_km": rng.uniform(0, 12, n),
        "rain_mm": rng.choice([1.0, 1.5, 3.0], size=n),
    })


def find_csv():
    for base in (Path.cwd(), Path(__file__).resolve().parent):
        for name in CSV_CANDIDATES:
            candidate = base / name
            if candidate.exists():
                return candidate
    return None


def main_source(row):
    """Main suspected source, read from the drivers text (nearer source wins)."""
    drivers = str(row.get("drivers", ""))
    candidates = []

    if "industr" in drivers:
        candidates.append((row.get("d_industry_km", 1e9), "Mapped industry"))
    if "wastewater" in drivers:
        candidates.append((row.get("d_wastewater_km", 1e9), "Wastewater site"))

    if candidates:
        return min(candidates, key=lambda c: (np.inf if pd.isna(c[0]) else c[0]))[1]
    if "river" in drivers:
        return "River/drain pathway"
    return "No dominant source flagged"


def recommend(row):
    tier = row.get("tier")
    confidence = row.get("confidence", "Unknown")

    action = TIER_ACTIONS.get(
        tier, "Review this location and verify with field testing."
    )

    exceedances = row.get("chem_exceedances")
    if isinstance(exceedances, str) and exceedances.strip():
        action += (
            f" Existing lab data show exceedances ({exceedances}); "
            "confirm with repeat testing."
        )

    if confidence == "Low":
        action += " Verify the input data because uncertainty is high."

    return action


def adapt(raw):
    """Convert the raw results CSV into the columns the dashboard uses."""
    df = raw.copy()

    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    df["type"] = df["type"].astype(str).str.title()
    df["risk"] = pd.to_numeric(df["risk"], errors="coerce")
    df["risk_score"] = (df["risk"] * 100).round(1)
    df["uncertainty"] = (
        pd.to_numeric(df["uncertainty"], errors="coerce") * 100
    ).round(1)

    # The CSV's own "priority" (0-1) becomes a score; "priority" is then
    # reused as the P1/P2 label derived from the model's tier.
    if "priority" in df.columns:
        df["priority_score"] = (
            pd.to_numeric(df["priority"], errors="coerce") * 100
        ).round(0)
    df["priority"] = df["tier"].map(TIER_TO_PRIORITY).fillna("Other")

    # Confidence bands relative to this dataset (thirds of the uncertainty)
    q1, q2 = df["uncertainty"].quantile([1 / 3, 2 / 3])
    df["confidence"] = np.where(
        df["uncertainty"].isna(), "Unknown",
        np.where(df["uncertainty"] <= q1, "High",
                 np.where(df["uncertainty"] <= q2, "Medium", "Low")),
    )

    df["drivers"] = df["drivers"].fillna("No dominant driver flagged")
    derived_source = df.apply(main_source, axis=1)
    if "main_source" in df.columns:
        df["main_source"] = df["main_source"].fillna(derived_source)
    else:
        df["main_source"] = derived_source

    known_zone = df["flood_status"].map({
        "Watch": "Flood watch zone",
        "Normal": "Outside flood zone",
    })
    other_zone = df["flood_status"].astype(object).where(
        df["flood_status"].notna(), "Unknown"
    )
    df["flood_zone"] = known_zone.fillna(other_zone).astype(str)

    # Real district exists only for lab-monitored wells; others show flood zone.
    # (map popups read the "district" column)
    df["district"] = [
        d.title() if isinstance(d, str) and d.strip() else zone
        for d, zone in zip(df["district"], df["flood_zone"])
    ]

    df["recommended_action"] = df.apply(recommend, axis=1)

    # Verified lab label (only for monitored wells; NaN elsewhere)
    label_source = df["contaminated"] if "contaminated" in df.columns else df["label_iron"]
    df["contaminated"] = to_binary(label_source)
    df["has_lab_data"] = df["station"].notna()

    # Highest risk score first everywhere in the app (model rank breaks ties)
    return df.sort_values(
        ["risk_score", "rank"], ascending=[False, True], na_position="last"
    ).reset_index(drop=True)


# ------------------------------------------------- upload / any-CSV support ----
ALIASES = {
    "lat": ["lat", "latitude", "y", "lat_deg"],
    "lon": ["lon", "lng", "long", "longitude", "x", "lon_deg"],
    "risk": ["risk", "risk_score", "risk_prob", "score", "probability",
             "prob", "contamination_risk"],
    "uncertainty": ["uncertainty", "uncertainty_pct", "unc", "std", "sigma"],
    "name": ["name", "location", "location_name", "site", "site_name",
             "village", "village_name", "station", "place"],
    "id": ["id", "site_id", "location_id", "uid", "code"],
    "type": ["type", "category", "kind", "location_type", "site_type"],
    "rank": ["rank", "priority_rank"],
    "tier": ["tier", "priority_tier", "action", "recommendation"],
    "label": ["contaminated", "label", "verified", "positive",
              "label_iron", "exceeds_limit"],
}
MAPPING_LABELS = {
    "lat": "Latitude *",
    "lon": "Longitude *",
    "risk": "Risk / score *",
    "name": "Name",
    "id": "ID",
    "type": "Type",
    "uncertainty": "Uncertainty",
    "rank": "Priority rank",
    "tier": "Testing tier",
    "label": "Verified label (0/1)",
}
REQUIRED_FIELDS = ("lat", "lon", "risk")
SAMPLE_CSV = (
    "name,lat,lon,risk,uncertainty\n"
    "Example village,26.30,91.50,0.72,0.25\n"
    "Example well,26.10,91.70,0.35,0.30\n"
)


@st.cache_data(show_spinner=False)
def read_csv_bytes(data):
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=encoding)
        except UnicodeDecodeError:
            continue
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            break
    return pd.DataFrame()


def to_binary(series):
    """Turn 0/1, yes/no, true/false style columns into 1.0 / 0.0 / NaN."""
    words = {"1": 1, "true": 1, "yes": 1, "y": 1, "positive": 1,
             "0": 0, "false": 0, "no": 0, "n": 0, "negative": 0}
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.where(numeric.isin([0, 1]), np.nan)
    text = series.astype(str).str.strip().str.lower().map(words)
    return numeric.fillna(text).astype(float)


def detect_columns(columns):
    """Guess which uploaded column plays which role, from common names."""
    normal = {str(c).strip().lower().replace(" ", "_"): c for c in columns}
    return {
        field: next((normal[a] for a in aliases if a in normal), None)
        for field, aliases in ALIASES.items()
    }


def scale_to_unit(series, label, notes):
    """Return values on a 0-1 scale (percentages and other ranges rescaled)."""
    values = pd.to_numeric(series, errors="coerce")
    top = values.max()

    if pd.isna(top) or top <= 1.0:
        return values
    if top <= 100.0:
        notes.append(f"{label}: values above 1 were read as percentages (0-100).")
        return values / 100.0

    low = values.min()
    notes.append(f"{label}: values were rescaled to 0-1 (min-max).")
    return (values - low) / (top - low) if top > low else values * 0.0


def prepare_upload(raw, mapping):
    """Build a frame in the app's standard format from any uploaded table."""
    notes = []
    out = raw.copy()
    out.columns = [str(c) for c in out.columns]
    src = {k: (str(v) if v is not None else None) for k, v in mapping.items()}

    out["lat"] = pd.to_numeric(raw[mapping["lat"]], errors="coerce")
    out["lon"] = pd.to_numeric(raw[mapping["lon"]], errors="coerce")
    out["risk"] = scale_to_unit(raw[mapping["risk"]], "Risk", notes)

    valid = (
        out["lat"].between(-90, 90)
        & out["lon"].between(-180, 180)
        & out["risk"].notna()
    )
    dropped = int((~valid).sum())
    if dropped:
        notes.append(
            f"{dropped} row(s) skipped: missing or invalid latitude, "
            "longitude or risk value."
        )
    out = out[valid].copy()
    raw_valid = raw.loc[out.index]
    out = out.reset_index(drop=True)
    raw_valid = raw_valid.reset_index(drop=True)

    # id (made unique so the field tracker works)
    if src["id"]:
        ids = raw_valid[mapping["id"]].astype(str)
    else:
        ids = pd.Series([f"ROW-{i + 1:04d}" for i in range(len(out))])
    repeat = ids.groupby(ids).cumcount()
    out["id"] = ids.where(repeat == 0, ids + "-" + repeat.astype(str))

    # name / type
    if src["name"]:
        out["name"] = raw_valid[mapping["name"]].astype(str)
        out["name"] = out["name"].where(out["name"].str.lower() != "nan", out["id"])
    else:
        out["name"] = [f"Location {i + 1}" for i in range(len(out))]
    out["type"] = (
        raw_valid[mapping["type"]].fillna("Location").astype(str)
        if src["type"] else "Location"
    )

    # uncertainty (optional)
    if src["uncertainty"]:
        out["uncertainty"] = scale_to_unit(
            raw_valid[mapping["uncertainty"]], "Uncertainty", notes
        )
    else:
        out["uncertainty"] = np.nan
        notes.append("No uncertainty column: confidence is shown as Unknown.")

    # rank / tier (computed from risk when not supplied)
    if src["rank"]:
        out["rank"] = pd.to_numeric(raw_valid[mapping["rank"]], errors="coerce")
    else:
        out["rank"] = out["risk"].rank(ascending=False, method="first")
        notes.append("Rank was computed from risk (highest risk = rank 1).")

    if src["tier"]:
        out["tier"] = raw_valid[mapping["tier"]].fillna("Unspecified").astype(str)
    else:
        cutoff = max(1, int(round(len(out) * 0.15)))
        out["tier"] = np.where(
            out["rank"].rank(method="first") <= cutoff,
            "Priority testing", "Routine testing",
        )
        notes.append("Tier was computed: top 15% by rank = Priority testing.")

    # verified label (optional)
    if src["label"]:
        out["contaminated"] = to_binary(raw_valid[mapping["label"]])

    return out, notes


def column_mapping_ui(raw, key):
    """Sidebar panel: auto-detected column roles the user can override."""
    detected = detect_columns(raw.columns)
    options = ["(none)"] + [str(c) for c in raw.columns]
    needs_help = any(detected[f] is None for f in REQUIRED_FIELDS)
    mapping = {}

    with st.sidebar.expander("Column mapping", expanded=needs_help):
        st.caption("Fields marked * are required. Detected automatically; "
                   "change them if a guess is wrong.")
        for field, label in MAPPING_LABELS.items():
            guess = str(detected[field]) if detected[field] is not None else None
            index = options.index(guess) if guess in options else 0
            choice = st.selectbox(label, options, index=index,
                                  key=f"map_{field}_{key}")
            mapping[field] = None if choice == "(none)" else choice

    return mapping


def load_data(uploaded):
    if uploaded is None:
        st.info("Upload a CSV file in the sidebar to view risk data.")
        st.stop()

    raw = read_csv_bytes(uploaded.getvalue())

    if raw.empty:
        st.error("The uploaded CSV is empty or could not be read.")
        st.stop()

    mapping = column_mapping_ui(raw, f"{uploaded.name}-{uploaded.size}")

    missing = [
        MAPPING_LABELS[field].rstrip(" *")
        for field in REQUIRED_FIELDS
        if mapping[field] is None
    ]

    if missing:
        st.error(
            "Choose the following columns in the sidebar "
            f"'Column mapping' panel: {', '.join(missing)}."
        )
        st.stop()

    prepared, notes = prepare_upload(raw, mapping)

    if prepared.empty:
        st.error(
            "No usable rows: every row had a missing or invalid "
            "latitude, longitude or risk value."
        )
        st.stop()

    return (
        adapt(prepared),
        False,
        f"uploaded file ({uploaded.name})",
        notes,
        [str(column) for column in raw.columns],
    )


def top_vs_random(df, frac=TOP_FRACTION, trials=500):
    """Compare the highest-risk share with random picks of the same size,
    using only locations that have a verified label (`contaminated`)."""
    if df.empty or "contaminated" not in df.columns:
        return None

    valid = df.dropna(subset=["contaminated", "risk_score"])

    if len(valid) < 5:
        return None

    n_select = max(1, int(len(valid) * frac))
    top_rate = valid.nlargest(n_select, "risk_score")["contaminated"].mean()

    rng = np.random.default_rng(7)
    values = valid["contaminated"].to_numpy()
    random_rates = [
        values[rng.choice(len(valid), size=n_select, replace=False)].mean()
        for _ in range(trials)
    ]

    return {
        "top_rate": float(top_rate),
        "random_rate": float(np.mean(random_rates)),
        "n_select": n_select,
        "n_labelled": int(len(valid)),
        "n_positive": int(valid["contaminated"].sum()),
    }


def explain(row):
    """Driver-based explanation: (summary sentence, driver list, factor table)."""
    drivers = [
        d.strip() for d in str(row.get("drivers", "")).split(";") if d.strip()
    ]

    rank = row.get("rank")
    rank_text = (
        f"ranked #{int(rank)} by the model" if pd.notna(rank)
        else "ranked by the model"
    )

    if drivers and drivers != ["No dominant driver flagged"]:
        message = (
            f"This location is {rank_text} mainly because: "
            f"{'; '.join(drivers)}."
        )
    else:
        message = (
            f"This location is {rank_text}, but no single dominant driver "
            "was flagged. Review the factor values below."
        )

    rows = [
        (label, round(float(row[col]), 1))
        for col, label in FACTOR_LABELS.items()
        if col in row.index and pd.notna(row[col])
    ]
    factors = pd.DataFrame(rows, columns=["Factor", "Value"])

    return message, drivers, factors


def lab_table(row):
    """Observed lab measurements for a monitored well (empty if none)."""
    rows = [
        (label, round(float(row[col]), 2))
        for col, label in LAB_LABELS.items()
        if col in row.index and pd.notna(row[col])
    ]
    return pd.DataFrame(rows, columns=["Parameter", "Value"])


def load_facilities(uploaded_file):
    if uploaded_file is None:
        return pd.DataFrame(columns=FACILITY_COLUMNS), None

    try:
        facilities = read_csv_bytes(uploaded_file.getvalue())
    except Exception:
        return pd.DataFrame(columns=FACILITY_COLUMNS), (
            "The uploaded CSV could not be read."
        )

    if facilities.empty:
        return pd.DataFrame(columns=FACILITY_COLUMNS), (
            "The uploaded CSV is empty or could not be read."
        )

    aliases = {
        "name": ["name", "facility_name", "hospital_name", "site_name",
                 "facility", "hospital", "institution"],
        "type": ["type", "facility_type", "hospital_type", "category",
                 "kind"],
        "district": ["district", "district_name", "admin_district"],
        "address": ["address", "full_address", "location", "place"],
        "latitude": ["latitude", "lat", "lat_deg", "y"],
        "longitude": ["longitude", "lon", "lng", "long", "lon_deg", "x"],
        "phone": ["phone", "telephone", "contact", "contact_number",
                  "phone_number", "mobile"],
        "hours": ["hours", "opening_hours", "working_hours", "timings"],
        "status": ["status", "facility_status", "operational_status"],
        "capacity": ["capacity", "bed_capacity", "beds", "available_beds"],
        "capacity_updated_at": ["capacity_updated_at", "capacity_updated",
                                "capacity_date"],
        "source": ["source", "data_source", "source_url"],
        "last_verified": ["last_verified", "verified_date", "last_updated"],
        "flood_status": ["flood_status", "flood_risk", "flood_zone"],
        "notes": ["notes", "remarks", "description"],
    }

    normalized = {
        str(column).strip().lower().replace(" ", "_").replace("-", "_"): column
        for column in facilities.columns
    }

    rename = {}

    for target, choices in aliases.items():
        for choice in choices:
            if choice in normalized:
                rename[normalized[choice]] = target
                break

    facilities = facilities.rename(columns=rename)

    for column in FACILITY_COLUMNS:
        if column not in facilities.columns:
            facilities[column] = pd.NA

    facilities["name"] = facilities["name"].fillna(
        facilities["type"].fillna("Unnamed facility")
    ).astype(str)

    for column in ["latitude", "longitude", "capacity"]:
        facilities[column] = pd.to_numeric(
            facilities[column], errors="coerce"
        )

    valid_coordinates = (
        facilities["latitude"].between(-90, 90)
        & facilities["longitude"].between(-180, 180)
    )

    facilities.loc[
        ~valid_coordinates, ["latitude", "longitude"]
    ] = np.nan

    facilities["district"] = (
        facilities["district"].astype("string").str.strip()
    )

    facilities = facilities.drop_duplicates().reset_index(drop=True)

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

    # OpenStreetMap needs no API key (CartoDB tiles now do).
    map_object = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="OpenStreetMap",
    )

    for _, facility in mapped.iterrows():
        name = str(facility["name"])
        facility_type = str(facility["type"])
        district = str(facility["district"])

        popup_lines = [
            f"<b>{html.escape(name)}</b>",
            f"Type: {html.escape(facility_type)}",
            f"District: {html.escape(district)}",
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
                    f"{label}: {html.escape(str(value))}"
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
    "rank",
    "name",
    "type",
    "flood_zone",
    "risk_score",
    "uncertainty",
    "confidence",
    "tier",
    "recommended_action",
]


def report_frame(sites):
    """Report/table view with scores shown as whole numbers."""
    frame = sites.reindex(columns=REPORT_COLS).copy()
    for column in ("risk_score", "uncertainty"):
        frame[column] = frame[column].map(
            lambda v: "" if pd.isna(v) else f"{v:.0f}"
        )
    # rows are ordered by risk, so label the CSV's rank as the model's rank
    return frame.rename(columns={"rank": "model_rank"})


def report_html(sites, n_total, demo):
    generated = datetime.now().strftime("%d %b %Y, %H:%M")

    table = report_frame(sites).to_html(
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
        <p>Locations included: {len(sites)} of {n_total} (highest risk score first)</p>
        {demo_note}
        {table}
        <p><em>Risk scores are model estimates, not confirmed contamination.
        Recommendations are advisory and subject to verification by
        qualified officials.</em></p>
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
    small = styles["BodyText"].clone("small", fontSize=7, leading=9)

    story = [
        Paragraph("AquaShield — Risk Report", styles["Title"]),
        Spacer(1, 10),
        Paragraph(
            f"Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}",
            styles["Normal"],
        ),
        Paragraph(
            f"Locations in report: {len(sites)} of {n_total} "
            "(highest risk score first)",
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

    report_data = report_frame(sites).fillna("")
    header = [str(c).replace("_", " ").title() for c in report_data.columns]
    body = [
        [Paragraph(html.escape(str(value)), small) for value in row]
        for row in report_data.values.tolist()
    ]

    table = Table(
        [header] + body,
        repeatRows=1,
        colWidths=[32, 85, 45, 85, 38, 50, 50, 75, 334],
    )

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
    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "<i>Risk scores are model estimates, not confirmed contamination. "
            "Recommendations are advisory and subject to verification by "
            "qualified officials.</i>",
            styles["Normal"],
        )
    )
    document.build(story)

    buffer.seek(0)
    return buffer.getvalue()


with st.sidebar:
    st.markdown("## 💧 AquaShield")
    st.caption("Flood & groundwater risk intelligence")
    st.divider()

    st.markdown("### Data source")
    uploaded_file = st.file_uploader(
        "Upload your own CSV (optional)",
        type=["csv"],
        help="Any CSV with latitude, longitude and a risk/score column. "
             "Remove the file to go back to the bundled data.",
    )

    with st.expander("Minimum CSV format"):
        st.write(
            "Required: **latitude, longitude, and a risk score** (0-1 or "
            "0-100). Optional: name, id, type, uncertainty, rank, tier, a "
            "0/1 verified label. Other columns are kept and shown under "
            "'All fields' in Risk Intelligence."
        )
        st.download_button(
            "⬇ Download example CSV",
            data=SAMPLE_CSV,
            file_name="aquashield_example.csv",
            mime="text/csv",
        )

df, is_demo, data_label, data_notes, source_columns = load_data(uploaded_file)

with st.sidebar:
    st.caption(f"Showing: {data_label} · {len(df):,} locations")
    for note in data_notes:
        st.caption(f"ℹ️ {note}")
    st.divider()

    st.markdown("### Dashboard filters")

    available_zones = sorted(
        df["flood_zone"].dropna().astype(str).unique()
    )

    selected_zones = st.multiselect(
        "Flood zone",
        options=available_zones,
        default=available_zones,
    )

    available_types = sorted(
        df["type"].dropna().astype(str).unique()
    )

    selected_types = st.multiselect(
        "Location type",
        options=available_types,
        default=available_types,
    )

    present_tiers = set(df["tier"].dropna().astype(str))
    available_tiers = [t for t in TIER_ORDER if t in present_tiers] + sorted(
        present_tiers - set(TIER_ORDER)
    )

    selected_tiers = st.multiselect(
        "Testing priority",
        options=available_tiers,
        default=available_tiers,
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
    df["flood_zone"].astype(str).isin(selected_zones)
    & df["type"].astype(str).isin(selected_types)
    & df["tier"].isin(selected_tiers)
    & (df["risk_score"].fillna(-1) >= min_risk)
].copy()

# Highest risk score first in every tab (model rank only breaks ties)
view = view.sort_values(
    ["risk_score", "rank"],
    ascending=[False, True],
    na_position="last",
).reset_index(drop=True)

priority_queue = view[
    view["priority"] == "P1"
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
        "not real field observations. Upload a CSV in the sidebar, or put "
        "risk_scores_priority_1.csv in the same folder as streamapp.py."
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

    n_labelled = int(view["contaminated"].notna().sum())
    n_positive = int(view["contaminated"].sum()) if n_labelled else 0

    metric1, metric2, metric3, metric4 = st.columns(4)

    metric1.metric(
        "Locations in view",
        f"{len(view):,}",
    )

    metric2.metric(
        "High risk (score ≥ 80)",
        f"{high_risk_count:,}",
    )

    metric3.metric(
        "Priority testing (P1)",
        f"{len(priority_queue):,}",
    )

    metric4.metric(
        "Verified positives (lab labels)",
        f"{n_positive} of {n_labelled}"
        if n_labelled
        else "No labels in view",
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
                    returned_objects=[],
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

        st.markdown(
            '<div class="priority-panel-title">Top priority sites &amp; actions</div>',
            unsafe_allow_html=True
        )

        if view.empty:
            st.info("No locations match the current filters.")
        else:
            top_sites = view.head(5)

            top_row = top_sites.iloc[0]
            top_score = top_row.get("risk_score")
            top_score_text = f"{top_score:.0f}" if pd.notna(top_score) else "N/A"
            top_name = html.escape(str(top_row.get("name", "Unnamed location")))
            top_zone = html.escape(str(top_row.get("flood_zone", "Unknown zone")))
            top_confidence = html.escape(str(top_row.get("confidence", "Unknown")))
            top_action = html.escape(str(top_row.get("recommended_action", "Review this location.")))

            if pd.notna(top_score) and top_score >= 80:
                alert_html = (
                    f'<div class="priority-alert"><strong>'
                    f'Alert: {top_name} ({top_zone}) - risk '
                    f'{top_score_text}/100, confidence {top_confidence}.'
                    f'</strong> {top_action}</div>'
                )
                st.markdown(alert_html, unsafe_allow_html=True)

            for _, row in top_sites.iterrows():
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

                item_html = (
                    f'<div class="priority-site">'
                    f'<div class="priority-site-heading">'
                    f'<span class="priority-site-dot">{dot}</span>'
                    f'<strong>{name}</strong>'
                    f'<span> · {score_text}/100 · confidence '
                    f'<strong>{confidence}</strong></span></div>'
                    f'<div class="priority-site-action">→ {action}</div></div>'
                )
                st.markdown(item_html, unsafe_allow_html=True)

            st.caption("Recommendations are advisory and require field verification.")

    st.divider()

    st.markdown("### Ranked sampling queue")
    st.caption(
        "Sorted by risk score, highest first. 'Model rank' is the rank "
        "supplied in the data file."
    )

    queue_columns = [
        "rank",
        "name",
        "type",
        "flood_zone",
        "risk_score",
        "uncertainty",
        "confidence",
        "main_source",
        "chem_exceedances",
        "tier",
        "recommended_action",
    ]

    st.dataframe(
        view.reindex(columns=queue_columns).dropna(axis=1, how="all"),
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": "Model rank",
            "flood_zone": "Flood zone",
            "risk_score": st.column_config.ProgressColumn(
                "Risk score", min_value=0, max_value=100, format="%.0f"
            ),
            "uncertainty": st.column_config.NumberColumn(
                "Uncertainty (%)", format="%.0f"
            ),
            "confidence": "Confidence",
            "main_source": "Main suspected source",
            "chem_exceedances": "Lab exceedances",
            "tier": "Testing priority",
            "recommended_action": "Recommended action",
        },
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
            "This check needs at least 5 locations with verified 0/1 labels "
            "in the current view. In the bundled data only the lab-monitored "
            "wells have labels, so keep 'Well' selected in the Location type "
            "filter. For an uploaded file, map a verified label column."
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric(
            f"Positive rate - highest-risk {comparison['n_select']}",
            f"{comparison['top_rate'] * 100:.0f}%",
        )
        c2.metric(
            f"Positive rate - random {comparison['n_select']}",
            f"{comparison['random_rate'] * 100:.0f}%",
        )
        c3.metric(
            "Labelled locations",
            f"{comparison['n_labelled']} ({comparison['n_positive']} positive)",
        )

        comparison_df = pd.DataFrame({
            "Selection": [
                "Highest-risk wells",
                "Random baseline",
            ],
            "Positive rate (%)": [
                comparison["top_rate"] * 100,
                comparison["random_rate"] * 100,
            ],
        }).set_index("Selection")

        st.bar_chart(comparison_df)

        st.caption(
            "Uses only the verified labels in the data (bundled file: "
            "iron-exceedance of lab-tested wells). With just "
            f"{comparison['n_labelled']} labelled locations this is a small, "
            "exploratory sample, not proof of field performance. If the "
            "model was trained using these same locations, it is not an "
            "independent test."
        )

with tab_why:
    st.subheader("Risk Intelligence")
    st.write(
        "Inspect the factors contributing to a location's recorded "
        "risk score. Locations are listed from highest to lowest risk."
    )

    if view.empty:
        st.info("No locations match the current filters.")
    else:
        choices = {
            f"{r['risk_score']:.0f}/100 · {r['name']} ({r['id']})": i
            for i, r in view.iterrows()
        }

        selected_label = st.selectbox(
            "Select location (type to search)",
            options=list(choices.keys()),
        )

        selected_row = view.loc[choices[selected_label]]

        detail1, detail2, detail3, detail4 = st.columns(4)

        score = selected_row.get("risk_score")
        uncertainty = selected_row.get("uncertainty")

        detail1.metric(
            "Risk score",
            f"{score:.0f}/100" if pd.notna(score) else "N/A",
        )

        detail2.metric(
            "Uncertainty",
            f"±{uncertainty:.0f}%" if pd.notna(uncertainty) else "N/A",
        )

        detail3.metric(
            "Confidence",
            selected_row.get("confidence", "Unknown"),
        )

        detail4.metric(
            "Testing priority",
            selected_row.get("tier", "Unranked"),
        )

        st.markdown("### Explanation")

        explanation, driver_list, factor_table = explain(selected_row)

        st.info(explanation)

        st.markdown(
            f"**Main suspected source:** {selected_row.get('main_source', 'Unknown')}"
        )

        if not factor_table.empty:
            st.markdown("### Measured factors")
            st.dataframe(
                factor_table,
                use_container_width=True,
                hide_index=True,
            )

        if bool(selected_row.get("has_lab_data")):
            st.markdown("### Observed lab measurements")

            n_samples = selected_row.get("n_samples")
            sample_text = (
                f" · {int(n_samples)} samples" if pd.notna(n_samples) else ""
            )
            st.caption(
                f"Monitoring station: {selected_row.get('station')} "
                f"({selected_row.get('district')}){sample_text}"
            )

            lab = lab_table(selected_row)
            if not lab.empty:
                st.dataframe(lab, use_container_width=True, hide_index=True)

            exceedances = selected_row.get("chem_exceedances")
            if isinstance(exceedances, str) and exceedances.strip():
                st.warning(f"Parameters flagged as exceeding limits: {exceedances}")
            else:
                st.success("No parameter exceedances recorded for this station.")

        with st.expander("All fields in the source file"):
            shown = [
                c for c in source_columns
                if c in selected_row.index and pd.notna(selected_row[c])
            ]
            st.dataframe(
                pd.DataFrame({
                    "Field": shown,
                    "Value": [str(selected_row[c]) for c in shown],
                }),
                use_container_width=True,
                hide_index=True,
            )

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
        "in the current view, highest risk first."
    )

    if "field_status" not in st.session_state:
        st.session_state.field_status = {}

    if view.empty:
        st.info("No locations match the current filters.")
    else:
        tracker = view[
            ["rank", "id", "name", "flood_zone", "risk_score", "tier"]
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
                "rank",
                "id",
                "name",
                "flood_zone",
                "risk_score",
                "tier",
            ],
            column_config={
                "rank": "Model rank",
                "risk_score": st.column_config.NumberColumn(
                    "Risk score", format="%.0f"
                ),
                "status": st.column_config.SelectboxColumn(
                    "Sampling status",
                    options=STATUSES,
                    required=True,
                ),
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
        max_value=max(2, len(view)),
        value=min(20, max(1, len(view))),
    )

    report_sites = view.head(report_count).copy()

    st.dataframe(
        report_frame(report_sites),
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
    st.write("Upload a facility CSV to view emergency services, locations, contact details and available capacity.")

    uploaded_facilities_file = st.file_uploader(
        "Upload emergency facilities CSV",
        type=["csv"],
        key="emergency_facilities_csv_upload"
    )

    if uploaded_facilities_file is None:
        st.info("Upload a CSV file to display emergency facilities.")
        st.caption("The facility map, directory and statistics will appear after a valid file is uploaded.")
    else:
        try:
            uploaded_facilities_file.seek(0)
            facilities = pd.read_csv(uploaded_facilities_file)

            facilities.columns = facilities.columns.str.strip().str.lower()

            column_aliases = {
                "latitude": ["latitude", "lat"],
                "longitude": ["longitude", "lon", "lng", "long"],
                "name": ["name", "facility_name", "hospital_name"],
                "type": ["type", "facility_type", "category"],
                "district": ["district", "district_name"],
                "address": ["address", "location"],
                "phone": ["phone", "contact", "contact_number", "phone_number"],
                "hours": ["hours", "opening_hours"],
                "status": ["status", "operational_status"],
                "capacity": ["capacity", "available_capacity", "beds"],
                "flood_status": ["flood_status"],
                "last_verified": ["last_verified"],
                "source": ["source"],
                "notes": ["notes"]
            }

            rename_columns = {}

            for standard_name, aliases in column_aliases.items():
                if standard_name not in facilities.columns:
                    for alias in aliases:
                        if alias in facilities.columns:
                            rename_columns[alias] = standard_name
                            break

            facilities = facilities.rename(columns=rename_columns)

            required_columns = ["name", "latitude", "longitude"]
            missing_columns = [
                column for column in required_columns
                if column not in facilities.columns
            ]

            if missing_columns:
                st.error(
                    "The uploaded CSV is missing required columns: "
                    + ", ".join(missing_columns)
                )
                st.write("Your CSV must contain facility name, latitude and longitude.")
                st.write("Available columns:", ", ".join(facilities.columns))
            elif facilities.empty:
                st.warning("The uploaded CSV contains no facility records.")
            else:
                for column in FACILITY_COLUMNS:
                    if column not in facilities.columns:
                        facilities[column] = pd.NA

                facilities["latitude"] = pd.to_numeric(
                    facilities["latitude"], errors="coerce"
                )
                facilities["longitude"] = pd.to_numeric(
                    facilities["longitude"], errors="coerce"
                )

                if "capacity" in facilities.columns:
                    facilities["capacity"] = pd.to_numeric(
                        facilities["capacity"], errors="coerce"
                    )

                facilities = facilities.dropna(
                    subset=["latitude", "longitude"]
                ).copy()

                facilities = facilities[
                    facilities["latitude"].between(-90, 90)
                    & facilities["longitude"].between(-180, 180)
                ].copy()

                if facilities.empty:
                    st.error(
                        "No valid coordinates were found. Check the latitude "
                        "and longitude values in your CSV."
                    )
                else:
                    active_mask = (
                        facilities["status"]
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .isin(["active", "open", "operational"])
                    )

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Facilities uploaded", len(facilities))
                    col2.metric("Marked operational", int(active_mask.sum()))
                    col3.metric(
                        "With capacity data",
                        int(facilities["capacity"].notna().sum())
                    )

                    st.divider()
                    st.markdown("### Facility map")
                    facility_map(facilities)

                    st.markdown("### Facility directory")
                    display_columns = [
                        "name", "type", "district", "address", "phone",
                        "hours", "status", "capacity", "flood_status",
                        "last_verified", "source", "notes"
                    ]

                    st.dataframe(
                        facilities.reindex(columns=display_columns),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.download_button(
                        "⬇ Download facility data",
                        data=facilities.to_csv(index=False).encode("utf-8"),
                        file_name="aquashield_emergency_facilities.csv",
                        mime="text/csv",
                        key="download_emergency_facilities"
                    )

        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as error:
            st.error(f"Could not read the uploaded CSV: {error}")
