import streamlit as st
import joblib
import numpy as np

st.write("App is running!")  # Debugging line

# Load the trained model
model = joblib.load('final_xgboost_model.pkl')

# Rest of your code...
