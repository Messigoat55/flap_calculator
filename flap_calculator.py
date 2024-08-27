import streamlit as st
import joblib
import numpy as np

st.write("App is running!")  # Confirm the app is starting

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

    st.write("Inputs received.")

    # Encode sex as numeric
    sex_encoded = 1 if sex == "Male" else 0

    # Create input array
    input_data = np.array([[age, sex_encoded, diabetes, cardiovascular, smoking, immunosuppression, albumin, prealbumin]])

    st.write("Input data prepared.")

# Predict probabilities of outcomes
if st.button("Predict Outcomes"):
    st.write("Predicting outcomes...")
    probabilities = model.predict_proba(input_data)

    outcomes = ["Infection", "Necrosis", "Congestion", "Seroma", "Hematoma", "Dehiscence", "Hospital Readmission"]
    
    # Display predicted probabilities
    st.subheader("Predicted Complication Probabilities:")
    for i, outcome in enumerate(outcomes):
        st.write(f"{outcome}: {probabilities[i][0][1] * 100:.2f}% chance")


