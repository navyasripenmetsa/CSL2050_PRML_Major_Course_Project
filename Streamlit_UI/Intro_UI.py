import streamlit as st
from streamlit_option_menu import option_menu
import requests #Added this line for HTTP requests to flask
from PIL import Image
import io
import svm_ui
import knn_ui
import rf_ui
import LightGBM_ui
import Naive_Bayes_ui
import Decision_Tree_UI
import Logistic_Regression_ui

# Background gradient styling for the main page
gradient_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #ff6600, #ff9900, #ffcc33, #ffdb4d);
    color : white;
}
</style>
"""
st.markdown(
    """
    <style>
    label {
        color: white !important;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(gradient_bg, unsafe_allow_html=True)

# Sidebar sky blue theme
sidebar_style = """
<style>
    [data-testid="stSidebar"] {
        background-color: #87CEEB;  /* Sky blue */
    }

    [data-testid="stSidebar"] .css-1cpxqw2 {
        color: #003366;  /* Dark blue text */
        font-weight: bold;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #003366;
    }

    .css-1v3fvcr.e1fqkh3o3 {
        background-color: #00BFFF !important;  /* Deep sky blue */
        color: white !important;
        font-weight: bold;
    }
</style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)

# Sidebar option menu
with st.sidebar:
    selected = option_menu(
        menu_title="SELECT MODEL",  # Sidebar title
        options=["HOME", "SVM", "KNN", "RF", "LIGHT-GBM", "NAIVE-BAYES","DECISION-TREE","LOGISTIC-REGRESSION"],
        menu_icon="cast",  # Optional icon
        default_index=0  # HOME is selected by default
    )
# Function to interact with Flask API (for predictions)
def get_prediction(image_data):
    try:
        # Convert image to bytes
        img_byte_array = io.BytesIO()
        image_data.save(img_byte_array, format='PNG')
        img_byte_array = img_byte_array.getvalue()
        
        # Send POST request with image to Flask API
        response = requests.post("http://127.0.0.1:5000/predict", files={"image": img_byte_array})
        
        if response.status_code == 200:
            return response.json()  # Return prediction
        else:
            return {"error": "Error in prediction request"}
    except Exception as e:
        return {"error": f"Exception occurred: {str(e)}"}

# Show content only for the selected model
if selected == "SVM":
    svm_ui.run()
elif selected == "KNN":
    knn_ui.run()
elif selected == "RF":
    rf_ui.run()
elif selected == "LIGHT-GBM":
    LightGBM_ui.run()
elif selected == "NAIVE-BAYES":
    Naive_Bayes_ui.run()
elif selected == "DECISION-TREE":
    Decision_Tree_UI.run()
elif selected == "LOGISTIC-REGRESSION":
    Logistic_Regression_ui.run()
else:  # HOME page content
    st.title("Pattern Recognition and Machine Learning Course Project (CSL2050) 📊")
    st.header("Project : Fruits Classification")
    st.divider()
    
     # User input for image upload
    uploaded_image = st.file_uploader("Upload an image of a fruit", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        # Open the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        
        if st.button("Predict"):
            # Get prediction from Flask backend
            result = get_prediction(image)
            
            if "result" in result:
                st.write(f"Prediction: {result['result']}")
            else:
                st.write(f"Error: {result.get('error', 'Unknown error')}")
    


    # Techniques Used
    st.markdown("**🤖 TECHNIQUES USED :**")
    st.markdown("**1. SVM (Support Vector Machines)**")
    st.markdown("**2. KNN (K-Nearest Neighbours)**")
    st.markdown("**3. RF (Random Forest)**")
    st.markdown("**4. LightGBM**")
    st.markdown("**5. Naive-Bayes**")
    st.markdown("**6. Decision-Trees**")
    st.markdown("**7. Logistic Regression**")
    st.divider()

    
    # Team Members
    st.markdown("**👨‍💻 DEVELOPED BY :**")
    st.markdown("**JADALA CHANDANA (B23CM1017)**")
    st.markdown("**MEEJURU LAKSHMI SOWMYA (B23CM1024)**")
    st.markdown("**PENMETSA NAVYASRI (B23CS1052)**")
    st.markdown("**GATTU CHARITHA (B23EE1021)**")
    st.markdown("**TUMMA SAI CHANDANA (B23EE1077)**")
    st.divider()

    # GitHub Link
    st.markdown("**🔗 GITHUB LINK:** [CSL2050_PRML_Major_Course_Project](https://github.com/navyasripenmetsa/CSL2050_PRML_Major_Course_Project)")
