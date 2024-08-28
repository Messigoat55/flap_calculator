import streamlit as st
import joblib
import numpy as np

# Set the page configuration with a centered layout and an icon
st.set_page_config(page_title="Leg Flap Risk Calculator", layout="centered", page_icon="🦵")

# Custom CSS for enhanced styling
st.markdown(
    """
    <style>
        /* Main styling */
        .main {
            background-color: #f0f4f8;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        /* Header and subheader styling */
        .header {
            color: #2c3e50;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 5px;
        }
        .subheader {
            color: #34495e;
            font-size: 24px;
            text-align: center;
            margin-bottom: 20px;
        }

        /* Button styling */
        .stButton>button {
            background-color: #2c3e50;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 5px;
            margin: 20px auto;
            display: block;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .stButton>button:hover {
            background-color: #1a252f;
        }

        /* Input and select box styling */
        .stNumberInput, .stSelectbox {
            margin-bottom: 10px;
        }
        input {
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 8px;
            font-size: 14px;
        }
        select {
            border: 1px solid #ccc;
            border-radius: 4px;
            padding: 8px;
            font-size: 14px;
        }

        /* Footer styling */
        .footer {
            font-size: 14px;
            text-align: center;
            margin-top: 20px;
            color: #95a5a6;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Display headers with the university name and calculator title
st.markdown('<div class="header">X University</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Leg Flap Risk Calculator</div>', unsafe_allow_html=True)

# Main app container
st.markdown('<div class="main">', unsafe_allow_html=True)

try:
    # Load the trained model
    st.info("Loading the model...")
    model = joblib.load('final_xgboost_model.pkl')
    st.success("Model loaded successfully.")
    
    # Input fields for patient characteristics
    st.header("Enter Patient Details")
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex", ["Male", "Female"])
    diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    cardiovascular = st.selectbox("Cardiovascular Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    smoking = st.selectbox("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    immunosuppression = st.selectbox("Immunosuppression", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
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
            st.write(f"**{outcome}**: {probabilities[i][0][1] * 100:.2f}% chance")

except Exception as e:
    st.error(f"An error occurred: {e}")

# Close the main container
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">© 2024 X University - All Rights Reserved</div>', unsafe_allow_html=True)
