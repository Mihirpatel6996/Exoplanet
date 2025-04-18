import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def show_dashboard(df_cumulative, df_stellar, df_merged):
    """
    Display the interactive dashboard page
    """
    st.header("📈 Interactive Analysis Dashboard")

    st.markdown("""
    This dashboard provides interactive visualizations for exploring the Kepler exoplanet dataset.
    Use the filters and controls to customize the visualizations and discover patterns in the data.
    """)

    # Create a sidebar for filters
    st.sidebar.subheader("Data Filters")

    # Filter by disposition
    dispositions = ['All'] + sorted(df_cumulative['koi_disposition'].unique().tolist())
    selected_disposition = st.sidebar.selectbox("Filter by Disposition", dispositions)

    # Filter by planet radius
    min_radius = float(df_cumulative['koi_prad'].min())
    max_radius = float(df_cumulative['koi_prad'].max())
    radius_range = st.sidebar.slider(
        "Planet Radius Range (Earth radii)",
        min_value=min_radius,
        max_value=max_radius,
        value=(min_radius, max_radius)
    )

    # Filter by orbital period
    min_period = float(df_cumulative['koi_period'].min())
    max_period = float(df_cumulative['koi_period'].max())
    period_range = st.sidebar.slider(
        "Orbital Period Range (days)",
        min_value=min_period,
        max_value=max_period,
        value=(min_period, max_period)
    )

    # Filter by equilibrium temperature
    min_temp = float(df_cumulative['koi_teq'].min())
    max_temp = float(df_cumulative['koi_teq'].max())
    temp_range = st.sidebar.slider(
        "Equilibrium Temperature Range (K)",
        min_value=min_temp,
        max_value=max_temp,
        value=(min_temp, max_temp)
    )

    # Apply filters to the data
    filtered_data = df_merged.copy()

    if selected_disposition != 'All':
        filtered_data = filtered_data[filtered_data['koi_disposition'] == selected_disposition]

    filtered_data = filtered_data[
        (filtered_data['koi_prad'] >= radius_range[0]) &
        (filtered_data['koi_prad'] <= radius_range[1]) &
        (filtered_data['koi_period'] >= period_range[0]) &
        (filtered_data['koi_period'] <= period_range[1]) &
        (filtered_data['koi_teq'] >= temp_range[0]) &
        (filtered_data['koi_teq'] <= temp_range[1])
    ]

    # Display metrics
    st.subheader("Dataset Overview")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric("Total Planets", len(filtered_data))

    with metric_col2:
        if 'koi_disposition' in filtered_data.columns:
            confirmed_count = (filtered_data['koi_disposition'] == 'CONFIRMED').sum()
            st.metric("Confirmed Planets", confirmed_count)
        else:
            st.metric("Confirmed Planets", 0)

    with metric_col3:
        if 'koi_disposition' in filtered_data.columns:
            candidate_count = (filtered_data['koi_disposition'] == 'CANDIDATE').sum()
            st.metric("Candidates", candidate_count)
        else:
            st.metric("Candidates", 0)

    with metric_col4:
        if 'koi_disposition' in filtered_data.columns:
            false_positive_count = (filtered_data['koi_disposition'] == 'FALSE POSITIVE').sum()
            st.metric("False Positives", false_positive_count)
        else:
            st.metric("False Positives", 0)

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Exoplanet Distribution", "Feature Relationships", "Stellar Properties", "Data Explorer"])

    # Exoplanet Distribution tab
    with tab1:
        st.subheader("Exoplanet Distribution")

        # Create a scatter plot of planet radius vs. orbital period
        scatter_options = st.multiselect(
            "Color by",
            options=['koi_disposition', 'koi_teq', 'koi_insol', 'teff', 'radius'],
            default=['koi_disposition']
        )

        color_by = scatter_options[0] if scatter_options else 'koi_disposition'

        # Create the scatter plot with Plotly
        fig = px.scatter(
            filtered_data,
            x='koi_period',
            y='koi_prad',
            color=color_by,
            hover_data=['kepid', 'koi_disposition', 'koi_teq', 'koi_insol'],
            title='Planet Radius vs. Orbital Period'
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

        # Create histograms of key features
        hist_col1, hist_col2 = st.columns(2)

        with hist_col1:
            # Create a histogram of planet radius
            fig_radius = px.histogram(
                filtered_data,
                x='koi_prad',
                nbins=30,
                title='Distribution of Planet Radius'
            )

            # Update layout
            fig_radius.update_layout(
                xaxis_title="Planet Radius (Earth radii)",
                yaxis_title="Count",
                width=400,
                height=300
            )

            # Display the plot
            st.plotly_chart(fig_radius)

        with hist_col2:
            # Create a histogram of orbital period
            fig_period = px.histogram(
                filtered_data,
                x='koi_period',
                nbins=30,
                title='Distribution of Orbital Period'
            )

            # Update layout
            fig_period.update_layout(
                xaxis_title="Orbital Period (days)",
                yaxis_title="Count",
                width=400,
                height=300
            )

            # Display the plot
            st.plotly_chart(fig_period)

        # Create a pie chart of dispositions
        if 'koi_disposition' in filtered_data.columns:
            disposition_counts = filtered_data['koi_disposition'].value_counts()

            # Create the pie chart with Plotly
            fig_pie = px.pie(
                values=disposition_counts.values,
                names=disposition_counts.index,
                title='Distribution of Dispositions'
            )

            # Update layout
            fig_pie.update_layout(
                width=500,
                height=500
            )

            # Display the chart
            st.plotly_chart(fig_pie)

    # Feature Relationships tab
    with tab2:
        st.subheader("Feature Relationships")

        # Create a correlation heatmap
        corr_cols = ['koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_steff', 'koi_srad', 'koi_smass',
                    'teff', 'logg', 'feh', 'mass', 'radius', 'dens', 'kepmag']

        # Filter out columns that don't exist in the dataframe
        corr_cols = [col for col in corr_cols if col in filtered_data.columns]

        # Calculate the correlation matrix
        corr_matrix = filtered_data[corr_cols].corr()

        # Create a heatmap with Plotly
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto='.2f',
            color_continuous_scale='RdBu_r',
            title='Correlation Matrix of Numerical Features'
        )

        # Update layout
        fig_heatmap.update_layout(
            width=800,
            height=800
        )

        # Display the heatmap
        st.plotly_chart(fig_heatmap)

        # Create a pairplot of selected features
        st.subheader("Feature Pairplot")

        # Select features for the pairplot
        pairplot_features = st.multiselect(
            "Select features for pairplot",
            options=corr_cols,
            default=['koi_period', 'koi_prad', 'koi_teq', 'koi_insol']
        )

        if pairplot_features and len(pairplot_features) >= 2:
            # Create a sample of the data for faster plotting
            sample_size = min(1000, len(filtered_data))
            sampled_data = filtered_data.sample(sample_size)

            # Create the scatter plot
            fig_pair = px.scatter(
                sampled_data,
                x=pairplot_features[0],
                y=pairplot_features[1],
                color='koi_disposition' if 'koi_disposition' in sampled_data.columns else None,
                title=f'{pairplot_features[1]} vs. {pairplot_features[0]}'
            )

            # Update layout
            fig_pair.update_layout(
                xaxis_title=pairplot_features[0],
                yaxis_title=pairplot_features[1],
                width=600,
                height=400
            )

            # Display the plot
            st.plotly_chart(fig_pair)

            # Create a 3D scatter plot if three features are selected
            if len(pairplot_features) >= 3:
                st.subheader("3D Scatter Plot")

                # Create the 3D scatter plot
                fig_3d = px.scatter_3d(
                    sampled_data,
                    x=pairplot_features[0],
                    y=pairplot_features[1],
                    z=pairplot_features[2],
                    color='koi_disposition' if 'koi_disposition' in sampled_data.columns else None,
                    title=f'3D Scatter Plot of {pairplot_features[0]}, {pairplot_features[1]}, and {pairplot_features[2]}'
                )

                # Update layout
                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title=pairplot_features[0],
                        yaxis_title=pairplot_features[1],
                        zaxis_title=pairplot_features[2]
                    ),
                    width=700,
                    height=500
                )

                # Display the plot
                st.plotly_chart(fig_3d)

    # Stellar Properties tab
    with tab3:
        st.subheader("Stellar Properties Analysis")

        # Create a scatter plot of stellar effective temperature vs. stellar radius
        fig_stellar = px.scatter(
            filtered_data,
            x='teff',
            y='radius',
            color='koi_disposition' if 'koi_disposition' in filtered_data.columns else None,
            hover_data=['kepid', 'koi_disposition', 'mass', 'logg', 'feh'],
            title='Stellar Radius vs. Effective Temperature (HR Diagram)'
        )

        # Update layout
        fig_stellar.update_layout(
            xaxis_title="Effective Temperature (K)",
            yaxis_title="Stellar Radius (Solar radii)",
            width=800,
            height=500
        )

        # Display the plot
        st.plotly_chart(fig_stellar)

        # Create histograms of stellar properties
        stellar_hist_col1, stellar_hist_col2 = st.columns(2)

        with stellar_hist_col1:
            # Create a histogram of stellar effective temperature
            fig_teff = px.histogram(
                filtered_data,
                x='teff',
                nbins=30,
                title='Distribution of Stellar Effective Temperature'
            )

            # Update layout
            fig_teff.update_layout(
                xaxis_title="Effective Temperature (K)",
                yaxis_title="Count",
                width=400,
                height=300
            )

            # Display the plot
            st.plotly_chart(fig_teff)

        with stellar_hist_col2:
            # Create a histogram of stellar radius
            fig_radius = px.histogram(
                filtered_data,
                x='radius',
                nbins=30,
                title='Distribution of Stellar Radius'
            )

            # Update layout
            fig_radius.update_layout(
                xaxis_title="Stellar Radius (Solar radii)",
                yaxis_title="Count",
                width=400,
                height=300
            )

            # Display the plot
            st.plotly_chart(fig_radius)

        # Create a scatter plot of stellar mass vs. stellar radius
        fig_mass_radius = px.scatter(
            filtered_data,
            x='mass',
            y='radius',
            color='koi_disposition' if 'koi_disposition' in filtered_data.columns else None,
            hover_data=['kepid', 'koi_disposition', 'teff', 'logg', 'feh'],
            title='Stellar Radius vs. Stellar Mass'
        )

        # Update layout
        fig_mass_radius.update_layout(
            xaxis_title="Stellar Mass (Solar masses)",
            yaxis_title="Stellar Radius (Solar radii)",
            width=800,
            height=500
        )

        # Display the plot
        st.plotly_chart(fig_mass_radius)

    # Data Explorer tab
    with tab4:
        st.subheader("Data Explorer")

        # Create a searchable table of the data
        st.dataframe(filtered_data)

        # Create a download button for the filtered data
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name="filtered_exoplanet_data.csv",
            mime="text/csv",
        )

        # Create a feature distribution explorer
        st.subheader("Feature Distribution Explorer")

        # Select a feature to explore
        feature_to_explore = st.selectbox(
            "Select a feature to explore",
            options=corr_cols
        )

        if feature_to_explore:
            # Create a histogram of the selected feature
            fig_feature = px.histogram(
                filtered_data,
                x=feature_to_explore,
                nbins=30,
                title=f'Distribution of {feature_to_explore}'
            )

            # Update layout
            fig_feature.update_layout(
                xaxis_title=feature_to_explore,
                yaxis_title="Count",
                width=800,
                height=400
            )

            # Display the plot
            st.plotly_chart(fig_feature)

            # Create a box plot of the selected feature by disposition
            if 'koi_disposition' in filtered_data.columns:
                fig_box = px.box(
                    filtered_data,
                    y=feature_to_explore,
                    x='koi_disposition',
                    title=f'Box Plot of {feature_to_explore} by Disposition'
                )

                # Update layout
                fig_box.update_layout(
                    xaxis_title="Disposition",
                    yaxis_title=feature_to_explore,
                    width=800,
                    height=400
                )

                # Display the plot
                st.plotly_chart(fig_box)

        # Create a custom scatter plot
        st.subheader("Custom Scatter Plot")

        # Select features for the x and y axes
        scatter_col1, scatter_col2, scatter_col3 = st.columns(3)

        with scatter_col1:
            x_feature = st.selectbox(
                "X-axis Feature",
                options=corr_cols,
                index=corr_cols.index('koi_period') if 'koi_period' in corr_cols else 0
            )

        with scatter_col2:
            y_feature = st.selectbox(
                "Y-axis Feature",
                options=corr_cols,
                index=corr_cols.index('koi_prad') if 'koi_prad' in corr_cols else 0
            )

        with scatter_col3:
            color_feature = st.selectbox(
                "Color by",
                options=['koi_disposition'] + corr_cols,
                index=0
            )

        # Create the custom scatter plot
        fig_custom = px.scatter(
            filtered_data,
            x=x_feature,
            y=y_feature,
            color=color_feature if color_feature in filtered_data.columns else None,
            hover_data=['kepid', 'koi_disposition'],
            title=f'{y_feature} vs. {x_feature}'
        )

        # Update layout
        fig_custom.update_layout(
            xaxis_title=x_feature,
            yaxis_title=y_feature,
            width=800,
            height=500
        )

        # Display the plot
        st.plotly_chart(fig_custom)
