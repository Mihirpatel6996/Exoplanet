import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta

def show_realtime_analysis(df_cumulative, df_stellar, df_merged, rf_model, xgb_model, scaler, feature_names):
    """
    Display the real-time analysis page
    """
    st.header("📊 Real-time Analysis System")

    if rf_model is None or xgb_model is None or scaler is None or feature_names is None:
        st.error("Models not loaded. Please run train_save_models.py first.")
        return

    st.markdown("""
    This page simulates real-time analysis of exoplanet data as it comes in from the Kepler telescope.
    You can adjust parameters to see how the system would process and analyze new observations.
    """)

    # Create a session state to store the simulated data
    if 'realtime_data' not in st.session_state:
        st.session_state.realtime_data = pd.DataFrame()
        st.session_state.last_update = datetime.now()
        st.session_state.update_count = 0

    # Sidebar controls for simulation
    st.sidebar.subheader("Simulation Controls")

    simulation_speed = st.sidebar.slider(
        "Simulation Speed",
        min_value=1,
        max_value=10,
        value=5,
        help="Controls how fast new data points are generated"
    )

    noise_level = st.sidebar.slider(
        "Noise Level",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.1,
        help="Controls the amount of noise added to the simulated data"
    )

    # Add a button to reset the simulation
    if st.sidebar.button("Reset Simulation"):
        st.session_state.realtime_data = pd.DataFrame()
        st.session_state.last_update = datetime.now()
        st.session_state.update_count = 0
        st.rerun()

    # Create tabs for different analysis views
    tab1, tab2, tab3 = st.tabs(["Live Feed", "Trend Analysis", "Anomaly Detection"])

    # Function to generate a new data point
    def generate_data_point():
        # Randomly select a row from the merged dataset
        random_idx = np.random.randint(0, len(df_merged))
        sample = df_merged.iloc[random_idx].copy()

        # Add some noise to the numerical columns
        for col in sample.index:
            if isinstance(sample[col], (int, float)) and col != 'kepid':
                sample[col] = sample[col] * (1 + noise_level * (np.random.random() - 0.5))

        # Create a dataframe with the sample
        sample_df = pd.DataFrame([sample])

        # Add a timestamp
        sample_df['timestamp'] = datetime.now()

        return sample_df

    # Function to make predictions on new data
    def predict_new_data(new_data):
        # Select only the columns used for prediction
        prediction_data = new_data.copy()

        # Handle missing values
        for col in prediction_data.select_dtypes(include=['number']).columns:
            if col in feature_names:
                prediction_data[col] = prediction_data[col].fillna(df_merged[col].median())

        # Feature engineering
        if 'koi_prad' in prediction_data.columns and 'radius' in prediction_data.columns:
            prediction_data['koi_prad_radius_ratio'] = prediction_data['koi_prad'] / prediction_data['radius']

        if 'koi_sma' in prediction_data.columns and 'radius' in prediction_data.columns:
            prediction_data['koi_sma_radius_ratio'] = prediction_data['koi_sma'] / prediction_data['radius']

        if 'koi_insol' in prediction_data.columns and 'koi_teq' in prediction_data.columns:
            prediction_data['koi_insol_teq_product'] = prediction_data['koi_insol'] * prediction_data['koi_teq']

        # Ensure the input data has the same columns as the training data
        missing_cols = set(feature_names) - set(prediction_data.columns)
        for col in missing_cols:
            prediction_data[col] = 0

        # Select only the columns used for prediction
        prediction_data = prediction_data[feature_names]

        # Scale the data
        prediction_data_scaled = scaler.transform(prediction_data)

        # Make predictions
        rf_pred_proba = rf_model.predict_proba(prediction_data_scaled)

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

        # Add the predictions to the data
        new_data['predicted_class'] = predicted_classes
        new_data['candidate_prob'] = candidate_probs
        new_data['confirmed_prob'] = confirmed_probs
        new_data['false_positive_prob'] = false_positive_probs

        return new_data

    # Check if it's time to update the data
    current_time = datetime.now()
    time_diff = (current_time - st.session_state.last_update).total_seconds()

    # Update the data if enough time has passed
    if time_diff > (11 - simulation_speed):
        # Generate a new data point
        new_data = generate_data_point()

        # Make predictions on the new data
        new_data_with_predictions = predict_new_data(new_data)

        # Add the new data to the session state
        if st.session_state.realtime_data.empty:
            st.session_state.realtime_data = new_data_with_predictions
        else:
            st.session_state.realtime_data = pd.concat([st.session_state.realtime_data, new_data_with_predictions], ignore_index=True)

        # Update the last update time
        st.session_state.last_update = current_time
        st.session_state.update_count += 1

    # Live Feed tab
    with tab1:
        st.subheader("Live Data Feed")

        # Display metrics
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        with metric_col1:
            st.metric("Data Points", len(st.session_state.realtime_data))

        with metric_col2:
            if not st.session_state.realtime_data.empty:
                confirmed_count = (st.session_state.realtime_data['predicted_class'] == 'CONFIRMED').sum()
                st.metric("Confirmed Exoplanets", confirmed_count)
            else:
                st.metric("Confirmed Exoplanets", 0)

        with metric_col3:
            if not st.session_state.realtime_data.empty:
                candidate_count = (st.session_state.realtime_data['predicted_class'] == 'CANDIDATE').sum()
                st.metric("Candidates", candidate_count)
            else:
                st.metric("Candidates", 0)

        with metric_col4:
            if not st.session_state.realtime_data.empty:
                false_positive_count = (st.session_state.realtime_data['predicted_class'] == 'FALSE POSITIVE').sum()
                st.metric("False Positives", false_positive_count)
            else:
                st.metric("False Positives", 0)

        # Display the live feed
        if not st.session_state.realtime_data.empty:
            # Create a container for the live feed
            live_feed_container = st.container()

            # Display the most recent data points
            with live_feed_container:
                # Sort by timestamp in descending order
                recent_data = st.session_state.realtime_data.sort_values('timestamp', ascending=False).head(10)

                # Convert probabilities to percentages for display
                display_df = recent_data[['kepid', 'koi_period', 'koi_prad', 'koi_teq', 'predicted_class',
                                         'candidate_prob', 'confirmed_prob', 'false_positive_prob', 'timestamp']].copy()

                # Format probabilities as percentages
                for prob_col in ['candidate_prob', 'confirmed_prob', 'false_positive_prob']:
                    display_df[prob_col] = display_df[prob_col].apply(lambda x: f"{float(x) * 100:.2f}%" if isinstance(x, (int, float)) else x)

                # Display the data
                st.dataframe(display_df)

            # Create a real-time visualization
            st.subheader("Real-time Visualization")

            # Create a scatter plot of planet radius vs. orbital period colored by prediction
            fig = px.scatter(
                st.session_state.realtime_data,
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
        else:
            st.info("Waiting for data... The simulation will generate new data points automatically.")

    # Trend Analysis tab
    with tab2:
        st.subheader("Trend Analysis")

        if len(st.session_state.realtime_data) > 5:
            # Group the data by timestamp (rounded to minutes)
            st.session_state.realtime_data['minute'] = st.session_state.realtime_data['timestamp'].dt.floor('min')
            grouped_data = st.session_state.realtime_data.groupby('minute')['predicted_class'].value_counts().unstack().fillna(0)

            # Reset the index to make 'minute' a column
            grouped_data = grouped_data.reset_index()

            # Ensure all classes are present
            for class_name in ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']:
                if class_name not in grouped_data.columns:
                    grouped_data[class_name] = 0

            # Create a line chart with Plotly
            fig = px.line(
                grouped_data,
                x='minute',
                y=['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'],
                color_discrete_map={'CONFIRMED': 'green', 'CANDIDATE': 'blue', 'FALSE POSITIVE': 'red'},
                title='Trend of Predictions Over Time'
            )

            # Update layout
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Count",
                width=800,
                height=400
            )

            # Display the chart
            st.plotly_chart(fig)

            # Create a stacked area chart with Plotly
            df_stacked = grouped_data.melt(id_vars=['minute'], value_vars=['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE'],
                                          var_name='Class', value_name='Count')

            fig = px.area(
                df_stacked,
                x='minute',
                y='Count',
                color='Class',
                color_discrete_map={'CONFIRMED': 'green', 'CANDIDATE': 'blue', 'FALSE POSITIVE': 'red'},
                title='Stacked Area Chart of Predictions Over Time'
            )

            # Update layout
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Count",
                width=800,
                height=400
            )

            # Display the plot
            st.plotly_chart(fig)

            # Display statistics
            st.subheader("Statistical Analysis")

            stat_col1, stat_col2 = st.columns(2)

            with stat_col1:
                # Calculate the mean and standard deviation of key features by class
                stats_df = st.session_state.realtime_data.groupby('predicted_class')[
                    ['koi_period', 'koi_prad', 'koi_teq']
                ].agg(['mean', 'std']).reset_index()

                # Display the statistics
                st.write("Feature Statistics by Class")
                st.dataframe(stats_df)

            with stat_col2:
                # Create a correlation heatmap
                corr_cols = ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol',
                            'candidate_prob', 'confirmed_prob', 'false_positive_prob']

                # Filter out columns that don't exist
                corr_cols = [col for col in corr_cols if col in st.session_state.realtime_data.columns]

                # Calculate correlation matrix
                corr_matrix = st.session_state.realtime_data[corr_cols].corr()

                # Create a heatmap with Plotly
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    color_continuous_scale='RdBu_r',
                    title='Correlation Matrix of Numerical Features'
                )

                # Update layout
                fig.update_layout(
                    width=600,
                    height=500
                )

                # Display the heatmap
                st.plotly_chart(fig)
        else:
            st.info("Not enough data for trend analysis. Wait for more data points to be generated.")

    # Anomaly Detection tab
    with tab3:
        st.subheader("Anomaly Detection")

        if len(st.session_state.realtime_data) > 5:
            # Define anomaly thresholds
            period_threshold = df_merged['koi_period'].quantile(0.95)
            radius_threshold = df_merged['koi_prad'].quantile(0.95)
            temp_threshold = df_merged['koi_teq'].quantile(0.95)

            # Detect anomalies
            anomalies = st.session_state.realtime_data[
                (st.session_state.realtime_data['koi_period'] > period_threshold) |
                (st.session_state.realtime_data['koi_prad'] > radius_threshold) |
                (st.session_state.realtime_data['koi_teq'] > temp_threshold)
            ]

            # Display anomaly metrics
            anomaly_col1, anomaly_col2, anomaly_col3 = st.columns(3)

            with anomaly_col1:
                st.metric("Total Anomalies", len(anomalies))

            with anomaly_col2:
                if not anomalies.empty:
                    period_anomalies = (anomalies['koi_period'] > period_threshold).sum()
                    st.metric("Period Anomalies", period_anomalies)
                else:
                    st.metric("Period Anomalies", 0)

            with anomaly_col3:
                if not anomalies.empty:
                    radius_anomalies = (anomalies['koi_prad'] > radius_threshold).sum()
                    st.metric("Radius Anomalies", radius_anomalies)
                else:
                    st.metric("Radius Anomalies", 0)

            # Display the anomalies
            if not anomalies.empty:
                st.subheader("Detected Anomalies")
                # Convert probabilities to percentages for display
                display_df = anomalies[['kepid', 'koi_period', 'koi_prad', 'koi_teq', 'predicted_class',
                                       'candidate_prob', 'confirmed_prob', 'false_positive_prob', 'timestamp']].copy()

                # Format probabilities as percentages
                for prob_col in ['candidate_prob', 'confirmed_prob', 'false_positive_prob']:
                    display_df[prob_col] = display_df[prob_col].apply(lambda x: f"{float(x) * 100:.2f}%" if isinstance(x, (int, float)) else x)

                # Display the data
                st.dataframe(display_df)

                # Create a scatter plot of anomalies
                fig = px.scatter(
                    anomalies,
                    x='koi_period',
                    y='koi_prad',
                    color='predicted_class',
                    color_discrete_map={'CANDIDATE': 'blue', 'CONFIRMED': 'green', 'FALSE POSITIVE': 'red'},
                    hover_data=['kepid', 'koi_teq'],
                    title='Anomalous Exoplanet Candidates'
                )

                # Add reference lines for thresholds
                fig.add_hline(y=radius_threshold, line_dash="dash", line_color="red", annotation_text="Radius Threshold")
                fig.add_vline(x=period_threshold, line_dash="dash", line_color="red", annotation_text="Period Threshold")

                # Update layout
                fig.update_layout(
                    xaxis_title="Orbital Period (days)",
                    yaxis_title="Planet Radius (Earth radii)",
                    width=800,
                    height=500
                )

                # Display the plot
                st.plotly_chart(fig)
            else:
                st.info("No anomalies detected yet.")
        else:
            st.info("Not enough data for anomaly detection. Wait for more data points to be generated.")

    # Auto-refresh the page
    st.empty()
    time.sleep(1)
    st.rerun()
