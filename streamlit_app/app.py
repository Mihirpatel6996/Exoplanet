import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Add parent directory to path to import from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set page configuration
st.set_page_config(
    page_title="Exoplanet Analysis Dashboard",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions to load data
@st.cache_data
def load_data():
    """Load and cache the datasets"""
    df_cumulative = pd.read_csv('../cumulative_2024.10.04_10.09.03.csv')
    df_stellar = pd.read_csv('../kepler_stellar_data.csv')
    df_merged = pd.merge(df_cumulative, df_stellar, on='kepid', how='inner')
    return df_cumulative, df_stellar, df_merged

# Define function to load models
@st.cache_resource
def load_models():
    """Load and cache the trained models"""
    try:
        rf_model = joblib.load('../models/random_forest_model.joblib')
        xgb_model = joblib.load('../models/xgboost_model.joblib')
        scaler = joblib.load('../models/scaler.joblib')
        feature_names = joblib.load('../models/feature_names.joblib')
        return rf_model, xgb_model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please run train_save_models.py first.")
        return None, None, None, None

# App title and description
st.title("🪐 Exoplanet Analysis Dashboard")
st.markdown("""
This application provides tools for analyzing Kepler exoplanet data, making predictions on new candidates,
and visualizing patterns in the dataset.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["Home", "Model Prediction", "Real-time Analysis", "Interactive Dashboard"]
)

# Load data
with st.spinner("Loading data..."):
    df_cumulative, df_stellar, df_merged = load_data()

# Load models
with st.spinner("Loading models..."):
    rf_model, xgb_model, scaler, feature_names = load_models()

# Home page
if page == "Home":
    st.header("Welcome to the Exoplanet Analysis Dashboard")
    
    st.subheader("Dataset Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Cumulative Dataset Rows", df_cumulative.shape[0])
        st.metric("Stellar Dataset Rows", df_stellar.shape[0])
    
    with col2:
        st.metric("Merged Dataset Rows", df_merged.shape[0])
        st.metric("Features Available", df_merged.shape[1])
    
    st.subheader("Available Features")
    
    # Display dataset samples
    tab1, tab2, tab3 = st.tabs(["Cumulative Data", "Stellar Data", "Merged Data"])
    
    with tab1:
        st.dataframe(df_cumulative.head())
    
    with tab2:
        st.dataframe(df_stellar.head())
    
    with tab3:
        st.dataframe(df_merged.head())
    
    st.subheader("Application Features")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("### 🔮 Model Prediction")
        st.markdown("""
        - Input candidate exoplanet data
        - Get predictions from trained models
        - View prediction confidence
        """)
    
    with feature_col2:
        st.markdown("### 📊 Real-time Analysis")
        st.markdown("""
        - Analyze new data as it comes in
        - Track changes in predictions
        - Monitor key metrics
        """)
    
    with feature_col3:
        st.markdown("### 📈 Interactive Dashboard")
        st.markdown("""
        - Explore data with interactive visualizations
        - Filter and sort data
        - Discover patterns and relationships
        """)

# Model Prediction page
elif page == "Model Prediction":
    from model_prediction import show_prediction_page
    show_prediction_page(rf_model, xgb_model, scaler, feature_names)

# Real-time Analysis page
elif page == "Real-time Analysis":
    from realtime_analysis import show_realtime_analysis
    show_realtime_analysis(df_cumulative, df_stellar, df_merged, rf_model, xgb_model, scaler, feature_names)

# Interactive Dashboard page
elif page == "Interactive Dashboard":
    from interactive_dashboard import show_dashboard
    show_dashboard(df_cumulative, df_stellar, df_merged)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This dashboard was created to analyze Kepler exoplanet data and make predictions on exoplanet candidates.
    """
)
