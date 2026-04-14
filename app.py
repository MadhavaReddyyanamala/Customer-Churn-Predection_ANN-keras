import streamlit as st
import numpy as np
import joblib
import time
from tensorflow.keras.models import load_model

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSight AI · Predictor",
    page_icon="🔮",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
/* ── Root & Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #0b0f1a;
    color: #e2e8f5;
}
/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2.5rem 2rem 4rem; max-width: 760px; }
/* ── Hero Header ── */
.hero {
    text-align: center;
    padding: 2.8rem 1.5rem 2rem;
    background: linear-gradient(135deg, #0f1628 0%, #111827 60%, #0d1220 100%);
    border: 1px solid #1e2d4a;
    border-radius: 20px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,179,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; right: -40px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(168,85,247,0.10) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,179,255,0.12);
    border: 1px solid rgba(99,179,255,0.3);
    color: #63b3ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    padding: 0.3rem 1rem;
    border-radius: 50px;
    margin-bottom: 1rem;
    text-transform: uppercase;
}
.hero h1 {
    font-family: 'Space Mono', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    color: #f0f6ff;
    line-height: 1.2;
    margin: 0.4rem 0;
    letter-spacing: -0.02em;
}
.hero h1 span { color: #ffffff; }
.hero-sub {
    color: #7b8bad;
    font-size: 0.95rem;
    font-weight: 300;
    margin-top: 0.6rem;
}
/* ── Section Labels ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #4a6080;
    margin-bottom: 0.8rem;
    margin-top: 1.8rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, #1e2d4a, transparent);
}
/* ── Input Card ── */
.input-card {
    background: #111827;
    border: 1px solid #1e2d4a;
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1rem;
}
/* ── Streamlit widget overrides ── */
.stSlider [data-baseweb="slider"] { margin-top: 0.3rem; }
div[data-testid="stSlider"] > label,
div[data-testid="stNumberInput"] > label,
div[data-testid="stSelectbox"] > label {
    font-size: 0.82rem;
    font-weight: 500;
    color: #94a9c9;
    letter-spacing: 0.02em;
    margin-bottom: 0.3rem;
}
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-color: #cbd5e1 !important;
    border-radius: 10px !important;
    color: #0f172a !important;
}
div[data-baseweb="select"] * { color: #0f172a !important; }
div[data-baseweb="input"] input { color: #0f172a !important; background: #ffffff !important; }
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.15) !important;
}
/* ── Predict Button ── */
div[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: #fff;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    font-weight: 700;
    padding: 0.85rem 2rem;
    border: none;
    border-radius: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
    text-transform: uppercase;
    margin-top: 0.5rem;
    box-shadow: 0 4px 20px rgba(37,99,235,0.35);
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #2563eb, #3b82f6);
    transform: translateY(-1px);
    box-shadow: 0 6px 28px rgba(37,99,235,0.5);
}
div[data-testid="stButton"] > button:active { transform: translateY(0); }
/* ── Risk Result Card ── */
.result-card {
    border-radius: 16px;
    padding: 2rem;
    margin-top: 1.5rem;
    text-align: center;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}
.result-card.high {
    background: linear-gradient(135deg, #1a0a0a, #1f0d0d);
    border-color: #7f1d1d;
}
.result-card.low {
    background: linear-gradient(135deg, #020f0a, #041510);
    border-color: #14532d;
}
.result-icon { font-size: 2.8rem; margin-bottom: 0.5rem; }
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.result-label.high { color: #f87171; }
.result-label.low  { color: #4ade80; }
.result-prob {
    font-family: 'Space Mono', monospace;
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    margin: 0.3rem 0;
}
.result-prob.high { color: #f87171; }
.result-prob.low  { color: #4ade80; }
.result-msg {
    font-size: 0.9rem;
    color: #94a9c9;
    margin-top: 0.6rem;
    font-weight: 300;
}
/* ── Progress Bar custom ── */
.stProgress > div > div > div {
    border-radius: 50px;
    height: 10px;
}
/* ── Metric chips ── */
.chip-row {
    display: flex;
    gap: 0.7rem;
    justify-content: center;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.chip {
    background: rgba(255,255,255,0.05);
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 0.45rem 1rem;
    font-size: 0.78rem;
    color: #7b8bad;
    font-family: 'Space Mono', monospace;
}
.chip span { color: #e2e8f5; font-weight: 700; }
/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 3rem;
    font-size: 0.72rem;
    color: #2d3f5e;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)

# ─── Load Model & Scaler ────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = load_model("Models/final_model.keras")
    scaler = joblib.load("Models/final_scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# ─── Hero Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🔮 AI-Powered · v2.0</div>
    <h1>Churn<span>Sight</span></h1>
    <p class="hero-sub">Predict customer churn risk with deep learning precision</p>
</div>
""", unsafe_allow_html=True)

# ─── Input Section ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">01 · Customer Profile</div>', unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        tenure = st.slider("Tenure (Months)", min_value=1, max_value=72, value=12,
                           help="How long has the customer been with you?")
        monthly = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0,
                                   value=50.0, step=0.5,
                                   help="Customer's monthly billing amount")

    with col2:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"],
                                 help="Type of service contract")
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"],
                                 help="Internet service subscribed by the customer")

# ─── Mappings ───────────────────────────────────────────────────────────────────
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_map = {"DSL": 0, "Fiber optic": 1, "No": 2}

# ─── Predict Button ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">02 · Prediction</div>', unsafe_allow_html=True)

if st.button("⚡ Analyze Churn Risk"):

    with st.spinner("Running inference..."):
        time.sleep(0.6)  # subtle UX delay

    # Prepare & predict
    features        = np.array([[tenure, monthly, contract_map[contract], internet_map[internet]]])
    features_scaled = scaler.transform(features)
    prob            = float(model.predict(features_scaled, verbose=0)[0][0])

    # Risk tier
    is_high = prob > 0.5
    tier_class   = "high" if is_high else "low"
    tier_icon    = "🔴" if is_high else "🟢"
    tier_label   = "HIGH RISK" if is_high else "LOW RISK"
    tier_message = (
        "This customer shows strong churn signals. Consider proactive retention."
        if is_high else
        "This customer appears stable and satisfied. Keep up the engagement!"
    )

    # Result card
    st.markdown(f"""
    <div class="result-card {tier_class}">
        <div class="result-icon">{tier_icon}</div>
        <div class="result-label {tier_class}">{tier_label}</div>
        <div class="result-prob {tier_class}">{prob:.1%}</div>
        <div class="result-msg">{tier_message}</div>
    </div>
    """, unsafe_allow_html=True)

    # Progress bar
    st.markdown("<br>", unsafe_allow_html=True)
    st.progress(prob)

    # Input summary chips
    st.markdown(f"""
    <div class="chip-row">
        <div class="chip">Tenure <span>{tenure}mo</span></div>
        <div class="chip">Monthly <span>${monthly:.0f}</span></div>
        <div class="chip">Contract <span>{contract}</span></div>
        <div class="chip">Internet <span>{internet}</span></div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    CHURNSIGHT AI · Built with Streamlit + TensorFlow · &copy; 2025
</div>
""", unsafe_allow_html=True)