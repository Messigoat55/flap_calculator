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
# Classy "Harvard-like" styling
# -----------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: #fbfbfc;
        color: #111827;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 2.2rem;
        max-width: 980px;
    }

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
        color: #0f172a;
        letter-spacing: -0.3px;
    }

    .hero-accent {
        color: #8C1515;
        font-weight: 700;
    }

    .hero-subtitle {
        margin-top: 8px;
        color: rgba(15,23,42,0.72);
        font-size: 15px;
    }

    .disclaimer {
        margin-top: 10px;
        color: rgba(15,23,42,0.55);
        font-size: 12px;
    }

    .card {
        background: #ffffff;
        border: 1px solid rgba(17,24,39,0.08);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(17,24,39,0.06);
        margin-bottom: 18px;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
    }

    .stFormSubmitButton > button {
        width: 100%;
        border-radius: 12px;
        padding: 0.7rem;
        background: #8C1515;
        color: white;
        font-weight: 700;
        border: none;
    }

    .results-title {
        font-family: ui-serif, Georgia, Cambria, "Times New Roman", Times, serif;
        font-size: 20px;
        margin-bottom: 10px;
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
        <div class="hero-subtitle">
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
# Main logic
# -----------------------------
try:
    model = joblib.load("final_xgboost_model.pkl")

    st.markdown('<div class="card">', unsafe_allow_html=True)

    with st.form("input_form"):
        st.markdown("**Age**")
        age = st.number_input("", 20, 100, 50)

        st.markdown("**Sex**")
        sex = st.selectbox("", ["Male", "Female"])

        st.markdown("**Diabetes**")
        diabetes = st.selectbox("", ["No", "Yes"])

        st.markdown("**Cardiovascular Disease**")
        cardiovascular = st.selectbox("", ["No", "Yes"])

        st.markdown("**Smoking**")
        smoking = st.selectbox("", ["No", "Yes"])

        st.markdown("**Immunosuppression**")
        immunosuppression = st.selectbox("", ["No", "Yes"])

        st.markdown("**Albumin (g/dL)**")
        albumin = st.number_input("", 1.0, 5.0, 3.5)

        st.markdown("**Prealbumin (mg/dL)**")
        prealbumin = st.number_input("", 5.0, 50.0, 20.0)

        submit = st.form_submit_button("Predict Outcomes")

    st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        sex_encoded = 1 if sex == "Male" else 0

        input_data = np.array([[
            age,
            sex_encoded,
            1 if diabetes == "Yes" else 0,
            1 if cardiovascular == "Yes" else 0,
            1 if smoking == "Yes" else 0,
            1 if immunosuppression == "Yes" else 0,
            albumin,
            prealbumin,
        ]])

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
                    <div>{pct:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )

except Exception as e:
    st.error(f"An error occurred: {e}")
