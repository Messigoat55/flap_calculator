import streamlit as st
import joblib
import numpy as np

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Tibial Limb Salvage Free Flap Risk Calculator",
    layout="wide",
    page_icon="🩺",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    /* Base app */
    .stApp {
        background: #fbfbfc;
        color: #111827;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial;
    }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main container */
    .block-container {
        max-width: 980px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    /* Header card */
    .hero {
        background: #ffffff;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    .hero-byline {
        font-size: 13px;
        color: #64748b;
        margin-bottom: 6px;
    }

    .hero-title {
        font-family: Georgia, Cambria, "Times New Roman", Times, serif;
        font-size: 34px;
        line-height: 1.15;
        margin: 0;
    }

    .hero-accent {
        color: #8C1515;
        font-weight: 800;
    }

    .hero-sub {
        margin-top: 8px;
        color: #475569;
        font-size: 15px;
    }

    .disclaimer {
        margin-top: 10px;
        font-size: 12px;
        color: #64748b;
    }

    /* REAL form card (Streamlit form container) */
    div[data-testid="stForm"] {
        background: #ffffff;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        margin-top: 0 !important;
    }

    /* Field labels */
    .field-label {
        font-weight: 700;
        margin: 10px 0 4px 0;
        color: #111827;
    }
    .field-first {
        margin-top: 0;
    }

    /* Tight label-input spacing */
    div[data-testid="stSelectbox"],
    div[data-testid="stNumberInput"] {
        margin-top: -6px;
        margin-bottom: 14px;
    }

    /* Submit button */
    .stFormSubmitButton button {
        width: 100%;
        background: #8C1515;
        color: white;
        border-radius: 12px;
        font-weight: 800;
        padding: 0.75rem;
        border: none;
        margin-top: 12px;
    }

    /* Results */
    .results-title {
        margin-top: 22px;
        margin-bottom: 14px;
        font-family: Georgia, Cambria, "Times New Roman", Times, serif;
        font-size: 20px;
        color: #111827;
    }

    .result-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 14px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        margin-bottom: 8px;
    }

    .result-name {
        font-weight: 800;
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
        <div class="hero-byline">
            Created by <strong>Louis Massoud, MD</strong>
        </div>

        <div class="hero-title">
            Tibial Limb Salvage <span class="hero-accent">Free Flap</span> Risk Calculator
        </div>

        <div class="hero-sub">
            Gustilo IIIB/C reconstruction • ML-based complication risk estimates (prototype)
        </div>

        <div class="disclaimer">
            For research and demonstration only. Not for clinical decision-making.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# App logic
# -----------------------------
try:
    model = joblib.load("final_xgboost_model.pkl")

    with st.form("input_form"):
        st.markdown('<div class="field-label field-first">Age</div>', unsafe_allow_html=True)
        age = st.number_input("", 20, 100, 50, key="age")

        st.markdown('<div class="field-label">Sex</div>', unsafe_allow_html=True)
        sex = st.selectbox("", ["Male", "Female"], key="sex")

        st.markdown('<div class="field-label">Diabetes</div>', unsafe_allow_html=True)
        diabetes = st.selectbox("", ["No", "Yes"], key="diabetes")

        st.markdown('<div class="field-label">Cardiovascular Disease</div>', unsafe_allow_html=True)
        cardiovascular = st.selectbox("", ["No", "Yes"], key="cvd")

        st.markdown('<div class="field-label">Smoking</div>', unsafe_allow_html=True)
        smoking = st.selectbox("", ["No", "Yes"], key="smoking")

        st.markdown('<div class="field-label">Immunosuppression</div>', unsafe_allow_html=True)
        immunosuppression = st.selectbox("", ["No", "Yes"], key="immuno")

        st.markdown('<div class="field-label">Albumin (g/dL)</div>', unsafe_allow_html=True)
        albumin = st.number_input("", 1.0, 5.0, 3.5, key="albumin")

        st.markdown('<div class="field-label">Prealbumin (mg/dL)</div>', unsafe_allow_html=True)
        prealbumin = st.number_input("", 5.0, 50.0, 20.0, key="prealbumin")

        submit = st.form_submit_button("Predict Outcomes")

    if submit:
        input_data = np.array([[
            age,
            1 if sex == "Male" else 0,
            1 if diabetes == "Yes" else 0,
            1 if cardiovascular == "Yes" else 0,
            1 if smoking == "Yes" else 0,
            1 if immunosuppression == "Yes" else 0,
            albumin,
            prealbumin
        ]])

        probabilities = model.predict_proba(input_data)

        outcomes = [
            "Infection",
            "Necrosis",
            "Congestion",
            "Seroma",
            "Hematoma",
            "Dehiscence",
            "Hospital Readmission"
        ]

        st.markdown(
            '<div class="results-title">Predicted complication probabilities</div>',
            unsafe_allow_html=True
        )

        for i, outcome in enumerate(outcomes):
            pct = probabilities[i][0][1] * 100
            st.markdown(
                f"""
                <div class="result-row">
                    <div class="result-name">{outcome}</div>
                    <div>{pct:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

except Exception as e:
    st.error(f"An error occurred: {e}")
