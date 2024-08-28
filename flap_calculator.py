import streamlit as st
import joblib
import numpy as np

# Set the page configuration
st.set_page_config(page_title="Leg Flap Risk Calculator", layout="centered")

# Set background color and styling
st.markdown(
    """
    <style>
        .main {
            background-color: white;
        }
        .header {
            color: maroon;
            font-size: 32px;
            font-weight: bold;
            text-align: center;
        }
        .subheader {
            color: maroon;
            font-size: 24px;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Display headers with the university name and calculator title
st.markdown('<div class="header">X University</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Leg Flap Risk Calculator</div>', unsafe_allow_html=True)

# Divider line for visual separation
st.markdown("---")

try:
    # Load the trained model
    st.write("Loading the model...")
    model = joblib.load('final_xgboost_model.pkl')
    st.write("Model loaded successfully.")
    
    # Input fields for patient characteristics
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    diabetes = st.selectbox("Diabetes", [0, 1])
    cardiovascular = st.selectbox("Cardiovascular Disease", [0, 1])
    smoking = st.selectbox("Smoking", [0, 1])
    immunosuppression = st.selectbox("Immunosuppression", [0, 1])
    albumin = st.number_input("Albumin Level (g/dL)", min_value=1.0, max_value=5.0, value=3.5)
    prealbumin = st.number_input("Prealbumin Level (mg/dL)", min_value=5.0, max_value=50.0, value=20.0)

    # Encode sex as numeric
    sex_encoded = 1 if sex == "Male" else 0

    # Create input array
    input_data = np.array([[age, sex_encoded, diabetes, cardiovascular, smoking, immunosuppression, albumin, prealbumin]])

    # Predict probabilities of outcomes
    if st.button("Predict Outcomes"):
        st.write("Predicting outcomes...")
        probabilities = model.predict_proba(input_data)

        outcomes = ["Infection", "Necrosis", "Congestion", "Seroma", "Hematoma", "Dehiscence", "Hospital Readmission"]
        
        # Display predicted probabilities
        st.subheader("Predicted Complication Probabilities:")
        for i, outcome in enumerate(outcomes):
            st.write(f"{outcome}: {probabilities[i][0][1] * 100:.2f}% chance")
except Exception as e:
    st.write(f"An error occurred: {e}")
