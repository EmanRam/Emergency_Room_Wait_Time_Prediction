from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="ER Wait Time Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS (custom layout/cards/buttons only) ───────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=Nunito:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    background: #f4f6fb !important;
    font-family: 'Nunito', sans-serif !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1b2a4a !important;
    border-right: none !important;
    padding-top: 0 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small {
    font-family: 'Nunito', sans-serif !important;
}

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #8fa3c8 !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    text-align: left !important;
    padding: 13px 20px !important;
    border-radius: 12px !important;
    border: none !important;
    width: 100% !important;
    box-shadow: none !important;
    transition: background 0.18s, color 0.18s !important;
    cursor: pointer !important;
    letter-spacing: 0 !important;
    transform: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #ffffff !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Hide sidebar resize handle */
[data-testid="stSidebarResizeHandle"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
    margin: 12px 0 !important;
}

/* ── Main content ── */
.block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1080px !important;
}
[data-testid="stHeader"] {
    background: transparent !important;
}

/* ── Typography ── */
h1 {
    font-family: 'DM Serif Display', serif !important;
    color: #1b2a4a !important;
}
h2, h3 {
    font-family: 'Nunito', sans-serif !important;
    color: #1b2a4a !important;
    font-weight: 700 !important;
}

/* ── Metric boxes ── */
[data-testid="stMetric"] {
    background: white;
    border-radius: 14px;
    padding: 20px 24px !important;
    border: 1px solid #e8ecf4;
    box-shadow: 0 2px 12px rgba(27,42,74,0.06);
}
[data-testid="stMetricLabel"] {
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #8898aa !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Serif Display', serif !important;
    font-size: 32px !important;
    color: #1b2a4a !important;
}

/* ── Feature cards ── */
.feat-card {
    background: white;
    border-radius: 16px;
    padding: 24px 26px 20px;
    border: 1px solid #e8ecf4;
    box-shadow: 0 2px 12px rgba(27,42,74,0.05);
    margin-bottom: 16px;
}
.feat-card-title {
    font-size: 17px;
    font-weight: 800;
    letter-spacing: 0.2px;
    color: #2563eb;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 10px;
    border-bottom: 2px solid #e8f0fe;
    padding-bottom: 12px;
}
.feat-card-title span {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 32px;
    height: 32px;
    border-radius: 9px;
    background: #dbeafe;
    font-size: 16px;
}

/* ── Predict button ── */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    color: white !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    padding: 14px 0 !important;
    border-radius: 12px !important;
    border: none !important;
    width: 100% !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.35) !important;
    letter-spacing: 0.3px !important;
    transition: all 0.25s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 28px rgba(37,99,235,0.45) !important;
}

/* ── Slider accent ── */
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #4a90d9 !important;
    box-shadow: 0 2px 8px rgba(74,144,217,0.4) !important;
}
div[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stTickBar"] + div div {
    background: #4a90d9 !important;
}

/* ── Result ── */
.result-card {
    border-radius: 20px;
    padding: 40px 36px;
    text-align: center;
    margin: 24px 0;
    animation: popIn 0.4s cubic-bezier(0.175,0.885,0.32,1.275) both;
}
.result-card.green { background: linear-gradient(135deg,#f0fdf4,#dcfce7); border: 2px solid #86efac; }
.result-card.amber { background: linear-gradient(135deg,#fffbeb,#fef3c7); border: 2px solid #fcd34d; }
.result-card.red   { background: linear-gradient(135deg,#fff1f2,#ffe4e6); border: 2px solid #fca5a5; }

.result-badge {
    display: inline-block;
    border-radius: 50px;
    padding: 5px 18px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 16px;
}
.green .result-badge { background: #d1fae5; color: #065f46; }
.amber .result-badge { background: #fde68a; color: #78350f; }
.red   .result-badge { background: #fee2e2; color: #991b1b; }

.result-number {
    font-family: 'DM Serif Display', serif;
    font-size: 80px;
    line-height: 1;
    margin-bottom: 4px;
}
.green .result-number { color: #059669; }
.amber .result-number { color: #d97706; }
.red   .result-number { color: #dc2626; }

.result-label { font-size: 16px; color: #6b7280; font-weight: 500; }
.result-note  { font-size: 13px; color: #9ca3af; margin-top: 12px; }

/* ── Info boxes ── */
.info-box {
    background: white;
    border-radius: 14px;
    padding: 20px 22px;
    border: 1px solid #e8ecf4;
    margin-bottom: 14px;
    box-shadow: 0 1px 6px rgba(27,42,74,0.04);
}
.info-box h4 {
    font-size: 15px;
    font-weight: 700;
    color: #1b2a4a;
    margin-bottom: 6px;
}
.info-box p  {
    font-size: 13px;
    color: #6b7280;
    line-height: 1.65;
    margin: 0;
}

/* ── Step pills ── */
.step-pill {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    background: white;
    border-radius: 14px;
    padding: 18px 20px;
    border: 1px solid #e8ecf4;
    margin-bottom: 12px;
    box-shadow: 0 1px 6px rgba(27,42,74,0.04);
}
.step-num {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: #1b2a4a;
    color: white;
    font-weight: 800;
    font-size: 15px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.step-text h4 {
    font-size: 14px;
    font-weight: 700;
    color: #1b2a4a;
    margin-bottom: 4px;
}
.step-text p  {
    font-size: 13px;
    color: #6b7280;
    margin: 0;
    line-height: 1.5;
}

/* ── Divider ── */
.section-divider {
    height: 1px;
    background: #e8ecf4;
    margin: 28px 0;
}

/* ── Animations ── */
@keyframes popIn {
    from { opacity: 0; transform: scale(0.92); }
    to   { opacity: 1; transform: scale(1); }
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    border-radius: 12px !important;
    border: 1px solid #e8ecf4 !important;
    background: white !important;
}
</style>
""", unsafe_allow_html=True)

# ── Paths / constants ────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "ER Wait Time Dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "er_wait_model.pkl"
TARGET = "Total Wait Time (min)"

FEATURE_COLS = [
    "Hospital Name",
    "Region",
    "Day of Week",
    "Season",
    "Time of Day",
    "Urgency Level",
    "Nurse-to-Patient Ratio",
    "Specialist Availability",
    "Facility Size (Beds)",
    "hour",
    "day",
    "month",
    "day_of_week",
]

FINAL_MODEL_INFO = {
    "name": "Gradient Boosting (No-Leak)",
    "r2_test": 0.846,
    "mae_test": 19.27,
    "rmse_test": 26.73,
    "desc": "Final trustworthy model trained on Dataset 3 after removing leaked time-component features.",
}

# ── Helpers ──────────────────────────────────────────────
def get_season(month: int) -> str:
    if month in [12, 1, 2]:
        return "Winter"
    if month in [3, 4, 5]:
        return "Spring"
    if month in [6, 7, 8]:
        return "Summer"
    return "Fall"

def get_time_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 21:
        return "Evening"
    return "Night"

def closest_category(preferred_value: str, available_values) -> str:
    available = [str(v) for v in available_values]
    available_lower_map = {v.lower(): v for v in available}

    if preferred_value.lower() in available_lower_map:
        return available_lower_map[preferred_value.lower()]

    fall_aliases = ["fall", "autumn"]
    if preferred_value.lower() in fall_aliases:
        for alias in fall_aliases:
            if alias in available_lower_map:
                return available_lower_map[alias]

    for v in available:
        if v.lower().startswith(preferred_value.lower()[:3]):
            return v

    return available[0] if available else preferred_value

def safe_int_series_bounds(series: pd.Series, default_min: int, default_max: int, default_value: int):
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return default_min, default_max, default_value
    smin = int(clean.min())
    smax = int(clean.max())
    sval = int(round(clean.median()))
    sval = max(smin, min(sval, smax))
    return smin, smax, sval

# ── Data loader ───────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    if not CSV_PATH.exists():
        st.error(f"❌ '{CSV_PATH}' not found.")
        st.stop()

    df = pd.read_csv(CSV_PATH)

    required_cols = [
        "Hospital Name",
        "Region",
        "Visit Date",
        "Day of Week",
        "Season",
        "Time of Day",
        "Urgency Level",
        "Nurse-to-Patient Ratio",
        "Specialist Availability",
        "Facility Size (Beds)",
        TARGET,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(f"❌ Missing required columns in CSV: {missing}")
        st.stop()

    df["Visit Date"] = pd.to_datetime(df["Visit Date"], errors="coerce")
    df = df.dropna(subset=["Visit Date"]).copy()
    df = df.sort_values("Visit Date").reset_index(drop=True)
    return df

# ── Model loader ──────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Loading trained model...")
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"❌ '{MODEL_PATH}' not found.\n\n"
            "Export the final no-leak Gradient Boosting pipeline from your notebook using joblib.dump()."
        )
        st.stop()

    artifact = joblib.load(MODEL_PATH)

    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact["model"]
        artifact_feature_cols = artifact.get("feature_cols")
    else:
        model = artifact
        artifact_feature_cols = None

    return model, artifact_feature_cols

# ── Session state ─────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"

# ── Load shared resources ─────────────────────────────────
df_ref = load_data()
model, artifact_feature_cols = load_model()

if artifact_feature_cols is not None and list(artifact_feature_cols) != FEATURE_COLS:
    st.warning(
        "The feature columns saved in the model artifact do not exactly match the app's FEATURE_COLS. "
        "If predictions look wrong, re-export the model using the same no-leak feature order used here."
    )

hospital_region_map = (
    df_ref.groupby("Hospital Name")["Region"]
    .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    .to_dict()
)

hospital_defaults = (
    df_ref.groupby("Hospital Name")[["Facility Size (Beds)", "Nurse-to-Patient Ratio", "Specialist Availability"]]
    .median(numeric_only=True)
    .reset_index()
)

facility_min, facility_max, facility_default_global = safe_int_series_bounds(
    df_ref["Facility Size (Beds)"], 10, 200, 80
)
nurse_min, nurse_max, nurse_default_global = safe_int_series_bounds(
    df_ref["Nurse-to-Patient Ratio"], 1, 5, 3
)
spec_min, spec_max, spec_default_global = safe_int_series_bounds(
    df_ref["Specialist Availability"], 0, 10, 3
)

season_values = sorted(df_ref["Season"].dropna().astype(str).unique().tolist())
tod_values = sorted(df_ref["Time of Day"].dropna().astype(str).unique().tolist())
urgency_values = sorted(df_ref["Urgency Level"].dropna().astype(str).unique().tolist())

default_visit_date = df_ref["Visit Date"].median().date()
min_visit_date = df_ref["Visit Date"].min().date()
max_visit_date = df_ref["Visit Date"].max().date()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:28px 20px 20px;border-bottom:1px solid rgba(255,255,255,0.08);margin-bottom:20px;">
      <div style="font-family:'DM Serif Display',serif;font-size:24px;color:white;line-height:1.2;">
        🏥 ER Wait<br><span style="color:#63b3ed;">Time AI</span>
      </div>
      <div style="font-size:12px;color:#6b82a8;margin-top:6px;">Dataset 3 Predictor</div>
    </div>
    <div style="padding:0 8px 8px;font-size:11px;font-weight:700;letter-spacing:1.5px;
                color:#4a5e80;text-transform:uppercase;">Menu</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <style>
    div[data-testid="stSidebar"] div.nav-home button {{
        color: {'#63b3ed' if st.session_state.page=='home' else '#8fa3c8'} !important;
        background: {'rgba(99,179,237,0.16)' if st.session_state.page=='home' else 'transparent'} !important;
        border-left: {'3px solid #63b3ed' if st.session_state.page=='home' else 'none'} !important;
        padding-left: {'17px' if st.session_state.page=='home' else '20px'} !important;
    }}
    div[data-testid="stSidebar"] div.nav-pred button {{
        color: {'#63b3ed' if st.session_state.page=='predict' else '#8fa3c8'} !important;
        background: {'rgba(99,179,237,0.16)' if st.session_state.page=='predict' else 'transparent'} !important;
        border-left: {'3px solid #63b3ed' if st.session_state.page=='predict' else 'none'} !important;
        padding-left: {'17px' if st.session_state.page=='predict' else '20px'} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="nav-home">', unsafe_allow_html=True)
        if st.button("🏠  Home", key="nav_home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="nav-pred">', unsafe_allow_html=True)
        if st.button("🔮  Predict Wait Time", key="nav_pred", use_container_width=True):
            st.session_state.page = "predict"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="padding:0 8px;">
      <div style="font-size:11px;font-weight:700;letter-spacing:1.5px;color:#4a5e80;
                  text-transform:uppercase;margin-bottom:12px;">Model Info</div>
      <div style="font-size:13px;color:#6b82a8;line-height:2.0;">
        📊 5,000 patient records<br>
        📋 13 no-leak input features<br>
        🎯 Target: Total Wait Time<br>
        🤖 {FINAL_MODEL_INFO['name']}
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding:0 8px;font-size:11px;color:#3a4d6b;">
      Built with Streamlit &amp; Scikit-learn
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.page

# ══════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════
if page == "home":
    st.markdown("<h1 style='font-size:42px;margin-bottom:6px;'>ER Wait Time Predictor</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:16px;color:#6b7280;margin-bottom:32px;'>"
        "A Dataset 3 deployment that predicts total emergency-room wait time using the final no-leak model."
        "</p>",
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Patient Records", "5,000")
    with m2:
        st.metric("Test R²", f"{FINAL_MODEL_INFO['r2_test']:.3f}")
    with m3:
        st.metric("MAE", f"{FINAL_MODEL_INFO['mae_test']:.2f} min")
    with m4:
        st.metric("RMSE", f"{FINAL_MODEL_INFO['rmse_test']:.2f} min")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")

    with col_l:
        st.markdown(
            "<h3 style='font-size:22px;font-weight:800;color:#2563eb;margin-bottom:16px;'>⚠️ The Problem</h3>",
            unsafe_allow_html=True,
        )
        for icon, title, desc in [
            ("🚨", "Unpredictable Congestion", "ER departments need better forecasts to anticipate spikes in demand and patient backlog."),
            ("⏱️", "Delayed Service Flow", "Without credible wait estimates, staff cannot proactively adjust beds, specialists, and triage flow."),
            ("📉", "Patient Experience Risk", "Long uncertain waits reduce satisfaction and make overcrowding more difficult to manage."),
        ]:
            st.markdown(f"""
            <div class="info-box">
              <h4>{icon} {title}</h4>
              <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.markdown(
            "<h3 style='font-size:22px;font-weight:800;color:#2563eb;margin-bottom:16px;'>✅ Final Project-Aligned Solution</h3>",
            unsafe_allow_html=True,
        )
        for icon, title, desc in [
            ("🤖", "No-Leak Deployment", "Uses the final no-leak Gradient Boosting model trained on Dataset 3 operational features."),
            ("🧠", "Operational Inputs", "Predictions are based on urgency, staffing, facility size, hospital, and visit timing."),
            ("📊", "Trustworthy Metrics", f"Final held-out performance: R² = {FINAL_MODEL_INFO['r2_test']:.3f}, MAE = {FINAL_MODEL_INFO['mae_test']:.2f}, RMSE = {FINAL_MODEL_INFO['rmse_test']:.2f}."),
        ]:
            st.markdown(f"""
            <div class="info-box">
              <h4>{icon} {title}</h4>
              <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.markdown(
        "<h3 style='font-size:22px;font-weight:800;color:#2563eb;margin-bottom:4px;'>🔄 How This App Matches Dataset 3</h3>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    for step_num, title, desc in [
        ("1", "Dataset 3 Inputs Only", "The form uses the no-leak Dataset 3 feature set: hospital, region, visit timing, urgency, staffing, and facility size."),
        ("2", "Derived Time Features", "The user enters visit date and visit hour. Day of week, season, month, day, and numeric weekday are derived automatically."),
        ("3", "Leakage Removed", "Leaked variables such as Time to Registration, Time to Triage, and Time to Medical Professional are intentionally excluded."),
        ("4", "Single Deployed Model", "The app loads the final saved Gradient Boosting pipeline and uses it for live prediction."),
    ]:
        st.markdown(f"""
        <div class="step-pill">
          <div class="step-num">{step_num}</div>
          <div class="step-text">
            <h4>{title}</h4>
            <p>{desc}</p>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="feat-card" style="border-top:3px solid #d97706;">
          <div class="feat-card-title"><span>🤖</span> Deployed Model</div>
          <div style="font-weight:700;font-size:16px;color:#1b2a4a;margin-bottom:6px;">{FINAL_MODEL_INFO['name']}</div>
          <div style="font-size:13px;color:#6b7280;line-height:1.7;">
            {FINAL_MODEL_INFO['desc']}<br><br>
            <b>Held-out Test Performance</b><br>
            R² = {FINAL_MODEL_INFO['r2_test']:.3f} &nbsp;·&nbsp;
            MAE = {FINAL_MODEL_INFO['mae_test']:.2f} min &nbsp;·&nbsp;
            RMSE = {FINAL_MODEL_INFO['rmse_test']:.2f} min
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "This app uses the Dataset 3 no-leak feature set. "
        "Leaked variables such as Time to Registration, Time to Triage, "
        "and Time to Medical Professional are intentionally excluded."
    )

    if MODEL_PATH.exists():
        st.success(f"✅ Saved model found: `{MODEL_PATH}`")
    else:
        st.error(f"❌ Saved model not found: `{MODEL_PATH}`")

# ══════════════════════════════════════════════════════════
# PREDICT PAGE
# ══════════════════════════════════════════════════════════
elif page == "predict":
    st.markdown("<h1 style='font-size:38px;margin-bottom:4px;'>Wait Time Estimator</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:15px;color:#6b7280;margin-bottom:28px;'>"
        "Enter Dataset 3 operational inputs and get an instant total wait-time estimate from the final no-leak model."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <div class="feat-card">
      <div class="feat-card-title"><span>🤖</span> Deployed Model</div>
      <p style="margin:0;font-size:14px;color:#6b7280;line-height:1.8;">
        <b>{FINAL_MODEL_INFO['name']}</b><br>
        {FINAL_MODEL_INFO['desc']}<br><br>
        Test R² = {FINAL_MODEL_INFO['r2_test']:.3f} &nbsp;·&nbsp;
        MAE = {FINAL_MODEL_INFO['mae_test']:.2f} min &nbsp;·&nbsp;
        RMSE = {FINAL_MODEL_INFO['rmse_test']:.2f} min
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    # Section 1
    st.markdown("""<div class="feat-card"><div class="feat-card-title"><span>🏥</span> Hospital & Location</div>""", unsafe_allow_html=True)

    hospital_name = st.selectbox("Hospital Name", sorted(df_ref["Hospital Name"].dropna().astype(str).unique()))
    region = hospital_region_map.get(hospital_name, "")

    hospital_row = hospital_defaults[hospital_defaults["Hospital Name"] == hospital_name]
    if not hospital_row.empty:
        facility_default = int(round(hospital_row["Facility Size (Beds)"].iloc[0]))
        nurse_default = int(round(hospital_row["Nurse-to-Patient Ratio"].iloc[0]))
        spec_default = int(round(hospital_row["Specialist Availability"].iloc[0]))
    else:
        facility_default = facility_default_global
        nurse_default = nurse_default_global
        spec_default = spec_default_global

    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Region", value=region, disabled=True)
    with col2:
        facility_size = st.slider(
            "🛏️  Facility Size (Beds)",
            min_value=facility_min,
            max_value=facility_max,
            value=max(facility_min, min(facility_default, facility_max)),
            step=1,
            help="Total number of beds in the hospital facility",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Section 2
    st.markdown("""<div class="feat-card"><div class="feat-card-title"><span>🕐</span> Visit Timing</div>""", unsafe_allow_html=True)

    col3, col4 = st.columns(2)
    with col3:
        visit_date = st.date_input(
            "Visit Date",
            value=default_visit_date,
            min_value=min_visit_date,
            max_value=max_visit_date,
        )
    with col4:
        visit_hour = st.slider(
            "⏰  Visit Hour",
            min_value=0,
            max_value=23,
            value=10,
            format="%02d:00",
            help="Hour of the ER visit (0 = midnight, 12 = noon)",
        )

    visit_ts = pd.Timestamp(visit_date)
    day_name = visit_ts.day_name()
    month_val = int(visit_ts.month)
    day_val = int(visit_ts.day)
    day_of_week_num = int(visit_ts.dayofweek)

    season = closest_category(get_season(month_val), season_values)
    time_of_day = closest_category(get_time_of_day(visit_hour), tod_values)

    col5, col6, col7 = st.columns(3)
    with col5:
        st.text_input("Day of Week", value=day_name, disabled=True)
    with col6:
        st.text_input("Season", value=season, disabled=True)
    with col7:
        st.text_input("Time of Day", value=time_of_day, disabled=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Section 3
    st.markdown("""<div class="feat-card"><div class="feat-card-title"><span>🧑‍⚕️</span> Patient Information</div>""", unsafe_allow_html=True)

    urgency_level = st.selectbox(
        "🚨  Urgency Level",
        urgency_values,
        help="Clinical urgency classification of the patient",
    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Section 4
    st.markdown("""<div class="feat-card"><div class="feat-card-title"><span>👩‍⚕️</span> Staffing & Resources</div>""", unsafe_allow_html=True)

    col8, col9 = st.columns(2)
    with col8:
        nurse_ratio = st.slider(
            "👩‍⚕️  Nurse-to-Patient Ratio",
            min_value=nurse_min,
            max_value=nurse_max,
            value=max(nurse_min, min(nurse_default, nurse_max)),
            step=1,
            help="Higher values indicate more nursing capacity available relative to demand",
        )
    with col9:
        specialist_avail = st.slider(
            "🩺  Specialist Availability",
            min_value=spec_min,
            max_value=spec_max,
            value=max(spec_min, min(spec_default, spec_max)),
            step=1,
            help="Number of specialists available in the ER",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.info(
        "The app derives Day of Week, Season, Time of Day, month, day, and numeric weekday "
        "from the visit date and visit hour to keep predictions consistent with Dataset 3."
    )

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        predict = st.button("⚡  Predict Wait Time")

    if predict:
        input_df = pd.DataFrame([{
            "Hospital Name": hospital_name,
            "Region": region,
            "Day of Week": day_name,
            "Season": season,
            "Time of Day": time_of_day,
            "Urgency Level": urgency_level,
            "Nurse-to-Patient Ratio": int(nurse_ratio),
            "Specialist Availability": int(specialist_avail),
            "Facility Size (Beds)": int(facility_size),
            "hour": int(visit_hour),
            "day": int(day_val),
            "month": int(month_val),
            "day_of_week": int(day_of_week_num),
        }])[FEATURE_COLS]

        prediction = max(0, round(float(model.predict(input_df)[0])))

        if prediction < 30:
            cls, badge, emoji = "green", "Short Wait", "🟢"
        elif prediction < 90:
            cls, badge, emoji = "amber", "Moderate Wait", "🟡"
        else:
            cls, badge, emoji = "red", "Long Wait", "🔴"

        hrs = prediction // 60
        mins_rem = prediction % 60
        time_str = f"{hrs}h {mins_rem}m" if hrs > 0 else f"{mins_rem} minutes"

        _, res_col, _ = st.columns([1, 2, 1])
        with res_col:
            st.markdown(f"""
            <div class="result-card {cls}">
              <div class="result-badge">{emoji} {badge}</div>
              <div class="result-number">{prediction}</div>
              <div class="result-label">minutes total wait</div>
              <div class="result-note">{time_str} &nbsp;·&nbsp; Model: {FINAL_MODEL_INFO['name']}</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("📋 View Input Summary"):
            summary = pd.DataFrame({
                "Category": [
                    "Hospital", "Hospital",
                    "Timing", "Timing", "Timing", "Timing", "Timing", "Timing",
                    "Patient",
                    "Staffing", "Staffing"
                ],
                "Feature": [
                    "Hospital Name", "Region",
                    "Visit Date", "Visit Hour", "Day of Week", "Season", "Time of Day", "Month / Day",
                    "Urgency Level",
                    "Nurse-to-Patient Ratio", "Specialist Availability"
                ],
                "Value": [
                    hospital_name, region,
                    str(visit_date), f"{visit_hour:02d}:00", day_name, season, time_of_day, f"{month_val}/{day_val}",
                    urgency_level,
                    nurse_ratio, specialist_avail
                ]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)