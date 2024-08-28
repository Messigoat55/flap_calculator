import streamlit as st
import joblib
import numpy as np

# Set the page configuration with a wide layout, custom title, and custom icon
st.set_page_config(
    page_title="Leg Flap Risk Calculator",
    layout="wide",  # Sets layout to wide
    page_icon="your_favicon.ico",  # Custom icon file
    menu_items={
        "Get Help": "https://www.streamlit.io",
        "Report a Bug": "https://github.com/streamlit/streamlit/issues",
        "About": "This is a demo app for Streamlit."
    }
)

# Add custom CSS to style the header and inputs, and to hide GitHub badge and main menu
st.markdown(
    """
    <style>
    /* Hide the main menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Hide GitHub badges and other dynamic elements */
    .css-1v3fvcr, 
    .css-1j6bx3l, 
    .viewerBadge_container__1QSob,
    .stViewerBadge,
    .stApp > header {
        display: none !important; 
        visibility: hidden !important;
    }
    /* Center the header and style */
    .header {
        color: #b22222; /* Bright maroon color */
        font-size: 40px;
        text-align: center;
        margin-bottom: 5px;
        font-weight: bold;
    }
    .subheader {
        color: #ffffff; /* White color for the subheader */
        font-size: 18px;
        text-align: center;
        margin-bottom: 20px;
        font-style: italic;
    }
    /* Style the input boxes and titles */
    .input-box {
        padding: 5px;
        border: 1px solid #ccc;
        margin-bottom: 10px;
        border-radius: 5px;
    }
    .input-box label {
        margin-bottom: 0;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the centered calculator title and subtitle
st.markdown('<div class="header">Leg Flap Risk Calculator</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">(Powered by Machine Learning)</div>', unsafe_allow_html=True)

try:
    # Load the trained model without unnecessary indicators
    model = joblib.load('final_xgboost_model.pkl')
    
    # Create a styled input container with labels and inputs touching each other
    with st.form(key='input_form'):
        st.markdown('<div class="input-box"><label>Age</label>', unsafe_allow_html=True)
        age = st.number_input("", min_value=20, max_value=100, value=50, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-box"><label>Sex</label>', unsafe_allow_html=True)
        sex = st.selectbox("", ["Male", "Female"], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-box"><label>Diabetes</label>', unsafe_allow_html=True)
        diabetes = st.selectbox("", [0, 1], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-box"><label>Cardiovascular Disease</label>', unsafe_allow_html=True)
        cardiovascular = st.selectbox("", [0, 1], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-box"><label>Smoking</label>', unsafe_allow_html=True)
        smoking = st.selectbox("", [0, 1], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-box"><label>Immunosuppression</label>', unsafe_allow_html=True)
        immunosuppression = st.selectbox("", [0, 1], label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-box"><label>Albumin Level (g/dL)</label>', unsafe_allow_html=True)
        albumin = st.number_input("", min_value=1.0, max_value=5.0, value=3.5, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="input-box"><label>Prealbumin Level (mg/dL)</label>', unsafe_allow_html=True)
        prealbumin = st.number_input("", min_value=5.0, max_value=50.0, value=20.0, label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

        # Submit button for the form
        submit_button = st.form_submit_button(label="Predict Outcomes")

    # Encode sex as numeric
    sex_encoded = 1 if sex == "Male" else 0

    # Create input array
    input_data = np.array([[age, sex_encoded, diabetes, cardiovascular, smoking, immunosuppression, albumin, prealbumin]])

    # Predict probabilities of outcomes if form is submitted
    if submit_button:
        probabilities = model.predict_proba(input_data)

        outcomes = ["Infection", "Necrosis", "Congestion", "Seroma", "Hematoma", "Dehiscence", "Hospital Readmission"]
        
        # Display predicted probabilities
        st.subheader("Predicted Complication Probabilities:")
        for i, outcome in enumerate(outcomes):
            st.write(f"{outcome}: {probabilities[i][0][1] * 100:.2f}% chance")

except Exception as e:
    st.error(f"An error occurred: {e}")
