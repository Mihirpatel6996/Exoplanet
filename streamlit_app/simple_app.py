import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
import plotly.express as px

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

# Load data
with st.spinner("Loading data..."):
    df_cumulative, df_stellar, df_merged = load_data()

# Load models
with st.spinner("Loading models..."):
    rf_model, xgb_model, scaler, feature_names = load_models()

# Model Prediction Page
st.header("🔮 Model Deployment and Prediction System")

if rf_model is None or xgb_model is None or scaler is None or feature_names is None:
    st.error("Models not loaded. Please run train_save_models.py first.")
else:
    st.markdown("""
    This page allows you to input data for a potential exoplanet candidate and get predictions from our trained models.
    """)

    # Create tabs for different input methods
    input_tab1, input_tab2 = st.tabs(["Manual Input", "Upload Data"])

    with input_tab1:
        st.subheader("Enter Candidate Data")

        # Create columns for input fields
        col1, col2, col3 = st.columns(3)

        with col1:
            koi_period = st.number_input("Orbital Period (days)", min_value=0.0, value=10.0, step=0.1)
            koi_prad = st.number_input("Planet Radius (Earth radii)", min_value=0.0, value=2.0, step=0.1)
            koi_sma = st.number_input("Semi-Major Axis (AU)", min_value=0.0, value=0.1, step=0.01)
            koi_teq = st.number_input("Equilibrium Temperature (K)", min_value=0.0, value=800.0, step=10.0)
            koi_insol = st.number_input("Insolation Flux (Earth flux)", min_value=0.0, value=100.0, step=1.0)
            koi_steff = st.number_input("Stellar Effective Temperature (K)", min_value=0.0, value=5500.0, step=100.0)
            koi_srad = st.number_input("Stellar Radius (Solar radii)", min_value=0.0, value=1.0, step=0.1)

        with col2:
            koi_smass = st.number_input("Stellar Mass (Solar mass)", min_value=0.0, value=1.0, step=0.1)
            teff = st.number_input("Stellar Effective Temperature (K) from Stellar Data", min_value=0.0, value=5500.0, step=100.0)
            logg = st.number_input("Surface Gravity (log10(cm/s^2))", min_value=0.0, value=4.5, step=0.1)
            feh = st.number_input("Metallicity [Fe/H]", min_value=-5.0, max_value=5.0, value=0.0, step=0.1)
            mass = st.number_input("Stellar Mass from Stellar Data (Solar mass)", min_value=0.0, value=1.0, step=0.1)
            radius = st.number_input("Stellar Radius from Stellar Data (Solar radii)", min_value=0.0, value=1.0, step=0.1)
            dens = st.number_input("Stellar Density (g/cm^3)", min_value=0.0, value=1.0, step=0.1)

        with col3:
            kepmag = st.number_input("Kepler Magnitude", min_value=0.0, value=12.0, step=0.1)
            koi_fpflag_nt = st.selectbox("Not Transit-Like Flag", [0, 1], index=0)
            koi_fpflag_ss = st.selectbox("Stellar Eclipse Flag", [0, 1], index=0)
            koi_fpflag_co = st.selectbox("Centroid Offset Flag", [0, 1], index=0)
            koi_fpflag_ec = st.selectbox("Ephemeris Match Flag", [0, 1], index=0)

            # Add a unique ID for the candidate
            kepid = st.number_input("Kepler ID (optional)", min_value=0, value=0, step=1)

        # Create a button to make predictions
        predict_button = st.button("Make Prediction", type="primary")

        if predict_button:
            # Create a dataframe with the input values
            input_data = pd.DataFrame({
                'kepid': [kepid],
                'koi_fpflag_nt': [koi_fpflag_nt],
                'koi_fpflag_ss': [koi_fpflag_ss],
                'koi_fpflag_co': [koi_fpflag_co],
                'koi_fpflag_ec': [koi_fpflag_ec],
                'koi_period': [koi_period],
                'koi_prad': [koi_prad],
                'koi_sma': [koi_sma],
                'koi_teq': [koi_teq],
                'koi_insol': [koi_insol],
                'koi_steff': [koi_steff],
                'koi_srad': [koi_srad],
                'koi_smass': [koi_smass],
                'teff': [teff],
                'logg': [logg],
                'feh': [feh],
                'mass': [mass],
                'radius': [radius],
                'dens': [dens],
                'kepmag': [kepmag]
            })

            # Feature engineering
            input_data['koi_prad_radius_ratio'] = input_data['koi_prad'] / input_data['radius']
            input_data['koi_sma_radius_ratio'] = input_data['koi_sma'] / input_data['radius']
            input_data['koi_insol_teq_product'] = input_data['koi_insol'] * input_data['koi_teq']

            # Ensure the input data has the same columns as the training data
            missing_cols = set(feature_names) - set(input_data.columns)
            for col in missing_cols:
                input_data[col] = 0

            # Reorder columns to match the training data
            input_data = input_data[feature_names]

            # Scale the input data
            input_data_scaled = scaler.transform(input_data)

            # Make predictions
            rf_pred_proba = rf_model.predict_proba(input_data_scaled)
            xgb_pred_proba = xgb_model.predict_proba(input_data_scaled)

            # Print raw predictions for debugging
            st.write(f"RF: {rf_pred_proba}, XGB: {xgb_pred_proba}")

            # Use only Random Forest predictions for simplicity
            if isinstance(rf_pred_proba, list):
                rf_pred_proba = np.array(rf_pred_proba)

            # Get the probabilities for the first (and only) sample
            if hasattr(rf_pred_proba, 'shape') and len(rf_pred_proba.shape) > 1:
                probs = rf_pred_proba[0]
            else:
                probs = rf_pred_proba

            # Use these probabilities
            avg_pred_proba = probs

            # Get the class with the highest probability
            class_names = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
            predicted_class_idx = np.argmax(avg_pred_proba)
            predicted_class = class_names[predicted_class_idx]

            # Display the prediction results
            st.subheader("Prediction Results")

            # Display the predicted class
            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.markdown(f"### Predicted Class: {predicted_class}")

                # Display the probabilities
                st.markdown("### Prediction Probabilities")

                # Create a dataframe with the probabilities
                proba_data = []
                for i, class_name in enumerate(class_names):
                    if i < len(avg_pred_proba):
                        proba_data.append({'Class': class_name, 'Probability': avg_pred_proba[i]})
                    else:
                        proba_data.append({'Class': class_name, 'Probability': 0.0})
                proba_df = pd.DataFrame(proba_data)

                # Convert probabilities to percentages for display
                for i in range(len(proba_df)):
                    try:
                        prob_value = proba_df.loc[i, 'Probability']
                        if isinstance(prob_value, (int, float)):
                            proba_df.loc[i, 'Probability'] = f"{prob_value * 100:.2f}%"
                        elif hasattr(prob_value, 'shape'):
                            # Handle numpy arrays
                            if prob_value.shape == ():
                                # Scalar array
                                proba_df.loc[i, 'Probability'] = f"{float(prob_value) * 100:.2f}%"
                            else:
                                # Multi-dimensional array - take the first element
                                proba_df.loc[i, 'Probability'] = f"{float(prob_value.flat[0]) * 100:.2f}%"
                        else:
                            # Try to convert to float
                            proba_df.loc[i, 'Probability'] = f"{float(prob_value) * 100:.2f}%"
                    except Exception as e:
                        st.write(f"Error converting probability for class {proba_df.loc[i, 'Class']}: {e}")
                        st.write(f"Value type: {type(prob_value)}, Value: {prob_value}")
                        proba_df.loc[i, 'Probability'] = "0.00%"

                # Display the probabilities
                st.dataframe(proba_df)

            with result_col2:
                # Create a copy of the dataframe for plotting
                plot_df = proba_df.copy()

                # Convert percentage strings back to floats for plotting
                plot_df['Probability'] = plot_df['Probability'].apply(lambda x: float(x.strip('%')) / 100 if isinstance(x, str) else x)

                # Create a bar chart of the probabilities using Plotly
                fig = px.bar(
                    plot_df,
                    x='Class',
                    y='Probability',
                    title='Prediction Probabilities'
                )

                # Update layout
                fig.update_layout(
                    xaxis_title='Class',
                    yaxis_title='Probability',
                    width=600,
                    height=400
                )

                # Display the chart
                st.plotly_chart(fig)

            # Display interpretation
            st.subheader("Prediction Interpretation")

            if predicted_class == 'CONFIRMED':
                st.success("This candidate is predicted to be a confirmed exoplanet with high confidence.")
                st.markdown("""
                - The model predicts this is likely a real exoplanet
                - The features align with patterns seen in confirmed exoplanets
                - Further observation may be warranted to confirm this prediction
                """)
            elif predicted_class == 'CANDIDATE':
                st.info("This candidate requires further investigation.")
                st.markdown("""
                - The model is uncertain about the classification
                - Additional observations or data may be needed
                - Consider follow-up studies to gather more evidence
                """)
            else:  # FALSE POSITIVE
                st.error("This candidate is predicted to be a false positive.")
                st.markdown("""
                - The model predicts this is likely not an exoplanet
                - The signal may be due to other phenomena such as:
                  - Stellar variability
                  - Eclipsing binary
                  - Instrumental artifacts
                """)

    with input_tab2:
        st.subheader("Upload Candidate Data")

        # File uploader
        uploaded_file = st.file_uploader("Upload a CSV file with candidate data", type=["csv"])

        if uploaded_file is not None:
            # Read the uploaded file
            try:
                uploaded_data = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully! Found {len(uploaded_data)} candidates.")

                # Display the uploaded data
                st.dataframe(uploaded_data.head())

                # Check if the required columns are present
                required_cols = ['koi_period', 'koi_prad', 'koi_sma', 'koi_teq', 'koi_insol',
                                'koi_steff', 'koi_srad', 'koi_smass', 'teff', 'logg',
                                'feh', 'mass', 'radius', 'dens', 'kepmag']

                missing_required = [col for col in required_cols if col not in uploaded_data.columns]

                if missing_required:
                    st.error(f"Missing required columns: {', '.join(missing_required)}")
                else:
                    # Process the data
                    process_button = st.button("Process Candidates", type="primary")

                    if process_button:
                        # Feature engineering
                        processed_data = uploaded_data.copy()

                        # Add flag columns if they don't exist
                        for flag in ['koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec']:
                            if flag not in processed_data.columns:
                                processed_data[flag] = 0

                        # Add kepid if it doesn't exist
                        if 'kepid' not in processed_data.columns:
                            processed_data['kepid'] = range(1, len(processed_data) + 1)

                        # Feature engineering
                        processed_data['koi_prad_radius_ratio'] = processed_data['koi_prad'] / processed_data['radius']
                        processed_data['koi_sma_radius_ratio'] = processed_data['koi_sma'] / processed_data['radius']
                        processed_data['koi_insol_teq_product'] = processed_data['koi_insol'] * processed_data['koi_teq']

                        # Ensure the input data has the same columns as the training data
                        missing_cols = set(feature_names) - set(processed_data.columns)
                        for col in missing_cols:
                            processed_data[col] = 0

                        # Select only the columns used for prediction
                        prediction_data = processed_data[feature_names]

                        # Scale the data
                        prediction_data_scaled = scaler.transform(prediction_data)

                        # Make predictions
                        rf_pred_proba = rf_model.predict_proba(prediction_data_scaled)

                        # Print raw predictions for debugging
                        st.write(f"Batch RF: {rf_pred_proba}")

                        # Handle the case where rf_pred_proba is a list of arrays
                        if isinstance(rf_pred_proba, list):
                            # For each sample, we need to determine the class with highest probability
                            predicted_classes = []
                            candidate_probs = []
                            confirmed_probs = []
                            false_positive_probs = []

                            # Get the number of samples
                            num_samples = len(rf_pred_proba[0])

                            # For each sample
                            for i in range(num_samples):
                                # Get probabilities for each class for this sample
                                candidate_prob = rf_pred_proba[0][i][0]
                                confirmed_prob = rf_pred_proba[1][i][0]
                                false_positive_prob = rf_pred_proba[2][i][0]

                                # Store probabilities
                                candidate_probs.append(candidate_prob)
                                confirmed_probs.append(confirmed_prob)
                                false_positive_probs.append(false_positive_prob)

                                # Find the class with highest probability
                                probs = [candidate_prob, confirmed_prob, false_positive_prob]
                                class_idx = np.argmax(probs)
                                class_names = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
                                predicted_classes.append(class_names[class_idx])
                        else:
                            # Handle the case where rf_pred_proba is a numpy array
                            class_names = ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']
                            if hasattr(rf_pred_proba, 'shape') and len(rf_pred_proba.shape) > 1:
                                predicted_classes = [class_names[i] for i in np.argmax(rf_pred_proba, axis=1)]
                                candidate_probs = rf_pred_proba[:, 0]
                                confirmed_probs = rf_pred_proba[:, 1]
                                false_positive_probs = rf_pred_proba[:, 2] if rf_pred_proba.shape[1] > 2 else np.zeros(len(rf_pred_proba))
                            else:
                                predicted_classes = [class_names[np.argmax(rf_pred_proba)]]
                                candidate_probs = [rf_pred_proba[0]]
                                confirmed_probs = [rf_pred_proba[1]]
                                false_positive_probs = [rf_pred_proba[2]] if len(rf_pred_proba) > 2 else [0]

                        # Add the predictions to the original data
                        results_df = processed_data.copy()
                        results_df['predicted_class'] = predicted_classes

                        # Add probabilities
                        try:
                            # Add the probabilities we calculated earlier
                            results_df['candidate_prob'] = candidate_probs
                            results_df['confirmed_prob'] = confirmed_probs
                            results_df['false_positive_prob'] = false_positive_probs
                        except Exception as e:
                            st.error(f"Error adding probabilities: {e}. Using default values.")
                            results_df['candidate_prob'] = 0.0
                            results_df['confirmed_prob'] = 0.0
                            results_df['false_positive_prob'] = 0.0

                        # Display the results
                        st.subheader("Prediction Results")

                        # Convert probabilities to percentages for display
                        display_df = results_df[['kepid', 'predicted_class', 'candidate_prob', 'confirmed_prob', 'false_positive_prob']].copy()
                        display_df['candidate_prob'] = display_df['candidate_prob'].astype(float) * 100
                        display_df['confirmed_prob'] = display_df['confirmed_prob'].astype(float) * 100
                        display_df['false_positive_prob'] = display_df['false_positive_prob'].astype(float) * 100

                        display_df['candidate_prob'] = display_df['candidate_prob'].apply(lambda x: f"{x:.2f}%")
                        display_df['confirmed_prob'] = display_df['confirmed_prob'].apply(lambda x: f"{x:.2f}%")
                        display_df['false_positive_prob'] = display_df['false_positive_prob'].apply(lambda x: f"{x:.2f}%")

                        st.dataframe(display_df)

                        # Create a summary of the predictions
                        prediction_counts = pd.Series(predicted_classes).value_counts()

                        # Display the summary
                        st.subheader("Prediction Summary")

                        summary_col1, summary_col2 = st.columns(2)

                        with summary_col1:
                            # Display the counts
                            for class_name in class_names:
                                count = prediction_counts.get(class_name, 0)
                                percentage = count / len(predicted_classes) * 100
                                st.metric(f"{class_name}", f"{count} ({percentage:.1f}%)")

                        with summary_col2:
                            # Create a pie chart with Plotly
                            fig = px.pie(
                                values=prediction_counts.values,
                                names=prediction_counts.index,
                                title='Distribution of Predictions'
                            )

                            # Update layout
                            fig.update_layout(
                                width=500,
                                height=500
                            )

                            # Display the chart
                            st.plotly_chart(fig)

                        # Option to download the results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results as CSV",
                            data=csv,
                            file_name="exoplanet_predictions.csv",
                            mime="text/csv",
                        )

                        # Create interactive visualizations with Plotly
                        st.subheader("Interactive Visualization")

                        # Create a scatter plot of planet radius vs. orbital period colored by prediction
                        fig = px.scatter(
                            results_df,
                            x='koi_period',
                            y='koi_prad',
                            color='predicted_class',
                            color_discrete_map={'CANDIDATE': 'blue', 'CONFIRMED': 'green', 'FALSE POSITIVE': 'red'},
                            hover_data=['kepid', 'confirmed_prob', 'koi_teq'],
                            title='Planet Radius vs. Orbital Period by Prediction Class'
                        )

                        # Update layout
                        fig.update_layout(
                            xaxis_title="Orbital Period (days)",
                            yaxis_title="Planet Radius (Earth radii)",
                            width=800,
                            height=500
                        )

                        # Display the plot
                        st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Error processing file: {e}")
