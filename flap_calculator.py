import streamlit as st
import joblib
import numpy as np

# Set the page configuration with a centered layout and an icon
st.set_page_config(page_title="Leg Flap Risk Calculator", layout="centered", page_icon="🦵")

# Custom CSS for the design
st.markdown(
    """
    <style>
        .subheader {
            color: #800000; /* Maroon color for the subheader */
            font-size: 24px;
            text-align: center;
            margin-bottom: 20px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the calculator title
st.markdown('<div class="subheader">Leg Flap Risk Calculator</div>', unsafe_allow_html=True)

try:
    # Load the trained model without unnecessary indicators
    model = joblib.load('final_xgboost_model.pkl')
    
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
        probabilities = model.predict_proba(input_data)

        outcomes = ["Infection", "Necrosis", "Congestion", "Seroma", "Hematoma", "Dehiscence", "Hospital Readmission"]
        
        # Display predicted probabilities
        st.subheader("Predicted Complication Probabilities:")
        for i, outcome in enumerate(outcomes):
            st.write(f"{outcome}: {probabilities[i][0][1] * 100:.2f}% chance")

except Exception as e:
    st.error(f"An error occurred: {e}")
