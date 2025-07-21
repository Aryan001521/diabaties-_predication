import streamlit as st
from pages.split import run_split_model
from pages.k_fold import run_k_fold_model

st.set_page_config(page_title="Diabetes Prediction", layout="wide")

st.title("🩺 Diabetes Prediction App")

# Sidebar model selection
st.sidebar.subheader("Select Model")
model_option = st.sidebar.radio("Choose a method:", ["Split Method", "K-Fold Cross Validation"])

# Call selected model function
if model_option == "Split Method":
    run_split_model()

elif model_option == "K-Fold Cross Validation":
    run_k_fold_model()
