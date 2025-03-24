import streamlit as st
from streamlit_option_menu import option_menu
import base64
import os

# Import ML Model UIs
import svm_ui
import knn_ui
import rf_ui
import LightGBM_ui
import Naive_Bayes_ui
import Decision_Tree_UI
import Logistic_Regression_ui

# --- Convert file to Base64 ---
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# --- Get Absolute Paths ---
base_dir = os.path.dirname(os.path.abspath(__file__))
gif_path = os.path.join(base_dir, "123.gif")
image_path = os.path.join(base_dir, "image.png")

# --- Sidebar Toggle Switch ---
with st.sidebar:
    enable_gif = st.toggle("Enable GIF Background", value=True)

# --- Apply Background GIF (if enabled) ---
if enable_gif:
    base64_gif = get_base64(gif_path)
    gif_css = f"""
    <style>
    .stApp {{
        position: relative;
        background-image: url("data:image/gif;base64,{base64_gif}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.6);  /* Semi-transparent white overlay */
        z-index: 0;
    }}
    .stApp > * {{
        position: relative;
        z-index: 1;
    }}

    /* Optional: Add shadow to text for better contrast */
    h1, h2, h3, h4, h5, h6, p, li, label {{
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }}
    </style>
    """
    st.markdown(gif_css, unsafe_allow_html=True)

# --- Sidebar Background Image ---
base64_image = get_base64(image_path)
sidebar_css = f"""
<style>
[data-testid="stSidebar"] {{
    background-image: url("data:image/png;base64,{base64_image}");
    background-size: cover;
    background-repeat: no-repeat;
}}
</style>
"""
st.markdown(sidebar_css, unsafe_allow_html=True)

# --- Sidebar Menu ---
with st.sidebar:
    selected = option_menu(
        menu_title="SELECT MODEL", 
        options=["HOME", "SVM", "KNN", "RF", "LIGHT-GBM", "NAIVE-BAYES", "DECISION-TREE", "LOGISTIC-REGRESSION"],
        menu_icon="cast",
        default_index=0
    )

# --- Main App Routing ---
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
else:
    # Home Page
    st.title("Pattern Recognition and Machine Learning Course Project (CSL2050) 📊")
    st.header("Project : Fruits Classification")
    st.divider()

    st.markdown("🤖 **TECHNIQUES USED:**")
    st.markdown("* 1. SVM (Support Vector Machines)*")
    st.markdown("* 2. KNN (K-Nearest Neighbours)*")
    st.markdown("* 3. RF (Random Forest)*")
    st.markdown("* 4. LightGBM*")
    st.markdown("* 5. Naive-Bayes*")
    st.markdown("* 6. Decision-Trees*")
    st.markdown("* 7. Logistic Regression*")
    st.divider()

    st.markdown("👨‍💻 **DEVELOPED BY:**")
    st.markdown("* JADALA CHANDANA (B23CM1017)*")
    st.markdown("* MEEJURU LAKSHMI SOWMYA (B23CM1024)*")
    st.markdown("* PENMETSA NAVYASRI (B23CS1052)*")
    st.markdown("* GATTU CHARITHA (B23EE1021)*")
    st.markdown("* TUMMA SAI CHANDANA (B23EE1077)*")
    st.divider()

    st.markdown("🔗 **GITHUB LINK:** [Click Here](https://github.com/navyasripenmetsa/CSL2050_PRML_Major_Course_Project)")
