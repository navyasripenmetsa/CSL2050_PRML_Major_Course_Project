import streamlit as st
from streamlit_option_menu import option_menu
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
st.markdown(gradient_bg, unsafe_allow_html=True)


# Custom CSS to change selected option color to violet
# Custom CSS to change the selected option's color to violet
sidebar_style = """
<style>
/* Change Selected Option Color to Violet */
div[data-testid="stSidebarNav"] > ul > li[data-menuitemid].active {
    background-color: #6a0dad !important; /* Violet */
    color: white !important;
    font-weight: bold !important;
}

/* Change Hover Color */
div[data-testid="stSidebarNav"] > ul > li[data-menuitemid]:hover {
    background-color: #dcd0ff !important; /* Light Violet */
    color: #6a0dad !important;
}
</style>
"""

st.markdown(sidebar_style, unsafe_allow_html=True)
# Sidebar with Model Selection
with st.sidebar:
    selected = option_menu(
        menu_title="SELECT MODEL",  # Sidebar title
        options=["HOME", "SVM", "KNN", "RF", "LIGHT-GBM", "NAIVE-BAYES","DECISION-TREE","LOGISTIC-REGRESSION"],
        menu_icon="cast",  # Optional icon
        default_index=0  # HOME is selected by default
    )

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
elif selected=="DECISION-TREE":
    Decision_Tree_UI.run()
elif selected=="LOGISTIC-REGRESSION":
    Logistic_Regression_ui.run()
else:  # HOME page content
    # Title and Project Information
    st.title("Pattern Recognition and Machine Learning Course Project (CSL2050) 📊")
    st.header("Project : Fruits Classification")
    st.divider()

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
    st.markdown("**🔗 GITHUB LINK:**(https://github.com/navyasripenmetsa/CSL2050_PRML_Major_Course_Project)")



