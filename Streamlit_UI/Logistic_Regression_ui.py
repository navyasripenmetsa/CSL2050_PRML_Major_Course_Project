import streamlit as st
import requests
from PIL import Image
from io import BytesIO

def run():
    # Set Page Title
    st.title("Logistic Regression")
    st.header("Upload Image")

    # File Upload
    uploaded_file = st.file_uploader("CHOOSE AN IMAGE", type=["jpg", "jpeg", "png"])

    # OR Image URL Input
    image_url = st.text_input("OR ENTER AN IMAGE URL")

    image = None

    # If an image is uploaded
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="", use_container_width=True)

    # If an image URL is provided
    elif image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption="", use_container_width=True)
        except:
            st.error("Failed to load image from the URL. Please check the link.")

    # Centering the Predict Button & Fixing Styling
    st.markdown(
        """
        <style>
        div.stButton > button {
            color: black !important; /* Black Text */
            background-color: white !important; /* White Background */
            font-size: 18px !important; 
            font-weight: bold !important;
            width: 150px !important;
            border: 2px solid black !important; /* Black Border */
            border-radius: 8px !important;
        }
        div.stButton > button:hover {
            background-color: #f0f0f0 !important; /* Light Gray on Hover */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Centering the Button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Predict"):
            st.success("🔍 Prediction logic will be implemented here.")
