import streamlit as st
import numpy as np

st.set_page_config(page_title="Leg Flap Risk Calculator", layout="centered", page_icon="🦵")

# Basic styling for layout testing
st.markdown(
    """
    <style>
        .main {
            background-color: #f0f4f8;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
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
    </style>
    """,
    unsafe_allow_html=True
)

# Display headers to test basic loading
st.markdown('<div class="header">X University</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Leg Flap Risk Calculator</div>', unsafe_allow_html=True)

# Basic input fields to confirm loading without predictions
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex", ["Male", "Female"])

st.write("Basic app elements loaded. Now gradually add the model and prediction parts back.")
