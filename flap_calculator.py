import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Tibial Limb Salvage Free Flap Risk Calculator",
    layout="wide",
    page_icon="🩺",  # use emoji (reliable) or replace with an actual .ico filename that exists in your repo
)

# -----------------------------
# Classy "Harvard-like" styling
# -----------------------------
st.markdown(
    """
    <style>
    /* --- Base app --- */
    .stApp {
        background: #fbfbfc;  /* off-white */
        color: #111827;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
    }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* --- Layout spacing --- */
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2.2rem;
        max-width: 980px;
    }

    /* --- Hero header --- */
    .hero {
        background: #ffffff;
        border: 1px solid rgba(17,24,39,0.08);
        border-radius: 18px;
        padding: 22px 26px;
        box-shadow: 0 6px 18px rgba(17,24,39,0.06);
        margin-bottom: 18px;
    }
    .hero-title {
        font-family: ui-serif, Georgia, Cambria, "Times New Roman", Times, serif;
        font-size: 34px;
        line-height: 1.15;
        margin: 0;
        color: #0f172a;
        letter-spacing: -0.3px;
    }
    .hero-accent {
        color: #8C1515; /* Harvard Crimson-like */
        font-weight: 700;
    }
    .hero-subtitle {
        margin-top: 8px;
        margin-bottom: 0;
        color: rgba(15,23,42,0.72);
        font-size: 15px;
    }

    /* Small disclaimer */
    .disclaimer {
        margin-top: 10px;
        color: rgba(15,23,42,0.55);
        font-size: 12px;
    }

    /* --- Form card --- */
    .card {
        background: #ffffff;
        border: 1px solid rgba(17,24,39,0.08);
        border-radius: 18px;
        padding: 18px 18px 10px 18px;
        box-shadow: 0 6px 18px rgba(17,24,39,0.06);
        margin-bottom: 18px;
    }
    .card label {
        font-weight: 600;
        color: rgba(15,23,42,0.78);
    }

    /* --- Inputs (Streamlit components) --- */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        border-color: rgba(17,24,39,0.18) !important;
        background: #ffffff !important;
    }
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within {
        border-color: rgba(140,21,21,0.55) !important;
        box-shadow: 0 0 0 4px rgba(140,21,21,0.10) !important;
    }

    /* --- Button --- */
    .stButton > button, .stFormSubmitButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 0.7rem 1rem;
        border: 1px solid rgba(140,21,21,0.35);
        background: #8C1515;
        color: #ffffff;
        font-weight: 700;
        letter-spacing: 0.2px;
    }
    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: #7a1212;
        border-color: rgba(140,21,21,0.55);
    }

    /* --- Results styling --- */
    .results-title {
        margin-top: 10px;
        margin-bottom: 10px;
        font-family: ui-serif, Georgia, Cambria, "Times New Roman", Times, serif;
        font-size: 20px;
        color: #0f172a;
    }
    .result-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 12px;
        border: 1px solid rgba(17,24,39,0.08);
        border-radius: 12px;
        background: #ffffff;
        margin-bottom: 8px;
    }
    .result-name {
        font-weight: 700;
        color: #0f172a;
    }
    .result-val {
        font-variant-numeric: tabular-nums;
        color: rgba(15,23,42,0.80);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            Tibial Limb Salvage <span class="hero-accent">Free Flap</span> Risk Calculator
        </div>
        <p class="hero-subtitle">
            Gustilo IIIB/C reconstruction • ML-based complication risk estimates (prototype)
        </p>
        <p class="disclaimer">
            For research/demonstration only. Not for clinical decision-making.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

try:
    # Load the trained model
    model = joblib.load("final_xgboost_model.pkl")

    # Form card wrapper
    st.markdown('<div class="card">', unsafe_allow_html=True)

 with st.form(key="input_form"):
    age = st.number_input("Age", min_value=20, max_value=100, value=50, key="age")
    sex = st.selectbox("Sex", ["Male", "Female"], key="sex")
    diabetes = st.selectbox("Diabetes", [0, 1], key="diabetes")
    cardiovascular = st.selectbox("Cardiovascular Disease", [0, 1], key="cardiovascular")
    smoking = st.selectbox("Smoking", [0, 1], key="smoking")
    immunosuppression = st.selectbox("Immunosuppression", [0, 1], key="immunosuppression")
    albumin = st.number_input("Albumin (g/dL)", min_value=1.0, max_value=5.0, value=3.5, key="albumin")
    prealbumin = st.number_input("Prealbumin (mg/dL)", min_value=5.0, max_value=50.0, value=20.0, key="prealbumin")

    submit_button = st.form_submit_button(label="Predict Outcomes")

    st.markdown("</div>", unsafe_allow_html=True)

    # Encode sex as numeric
    sex_encoded = 1 if sex == "Male" else 0

    # Input array
    input_data = np.array([[age, sex_encoded, diabetes, cardiovascular, smoking, immunosuppression, albumin, prealbumin]])

    # Predict + display
    if submit_button:
        probabilities = model.predict_proba(input_data)

        outcomes = [
            "Infection",
            "Necrosis",
            "Congestion",
            "Seroma",
            "Hematoma",
            "Dehiscence",
            "Hospital Readmission",
        ]

        st.markdown('<div class="results-title">Predicted complication probabilities</div>', unsafe_allow_html=True)

        for i, outcome in enumerate(outcomes):
            pct = probabilities[i][0][1] * 100
            st.markdown(
                f"""
                <div class="result-row">
                    <div class="result-name">{outcome}</div>
                    <div class="result-val">{pct:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

except Exception as e:
    st.error(f"An error occurred: {e}")
