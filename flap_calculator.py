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
    .stApp {
        background: #fbfbfc;
        color: #111827;
        font-family: system-ui, -apple-system, BlinkMacSystemFont;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .block-container {
        max-width: 980px;
        padding-top: 2rem;
    }

    .hero {
        background: white;
        border-radius: 18px;
        padding: 24px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
        margin-bottom: 20px;
    }

    .hero-title {
        font-family: Georgia, serif;
        font-size: 34px;
    }

    .hero-accent {
        color: #8C1515;
        font-weight: bold;
    }

    .hero-sub {
        margin-top: 6px;
        color: #475569;
        font-size: 15px;
    }

    .disclaimer {
        margin-top: 10px;
        font-size: 12px;
        color: #64748b;
    }

    .card {
        background: white;
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.06);
    }

    .stFormSubmitButton button {
        width: 100%;
        background: #8C1515;
        color: white;
        border-radius: 12px;
        font-weight: 700;
        padding: 0.7rem;
        border: none;
    }

    .results-title {
        margin-top: 20px;
        font-family: Georgia, serif;
        font-size: 20px;
    }

    .result-row {
        display: flex;
        justify-content: space-between;
        padding: 10px 14px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: white;
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

    st.markdown('<div class="card">', unsafe_allow_html=True)

    with st.form("input_form"):

        st.markdown("**Age**")
        age = st.number_input("", 20, 100, 50, key="age")

        st.markdown("**Sex**")
        sex = st.selectbox("", ["Male", "Female"], key="sex")

        st.markdown("**Diabetes**")
        diabetes = st.selectbox("", ["No", "Yes"], key="diabetes")

        st.markdown("**Cardiovascular Disease**")
        cardiovascular = st.selectbox("", ["No", "Yes"], key="cvd")

        st.markdown("**Smoking**")
        smoking = st.selectbox("", ["No", "Yes"], key="smoking")

        st.markdown("**Immunosuppression**")
        immunosuppression = st.selectbox("", ["No", "Yes"], key="immuno")

        st.markdown("**Albumin (g/dL)**")
        albumin = st.number_input("", 1.0, 5.0, 3.5, key="albumin")

        st.markdown("**Prealbumin (mg/dL)**")
        prealbumin = st.number_input("", 5.0, 50.0, 20.0, key="prealbumin")

        submit = st.form_submit_button("Predict Outcomes")

    st.markdown("</div>", unsafe_allow_html=True)

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
