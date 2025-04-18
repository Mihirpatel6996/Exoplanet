# Exoplanet Analysis Dashboard

This project provides a comprehensive dashboard for analyzing Kepler exoplanet data, making predictions on new candidates, and visualizing patterns in the dataset.

## Features

### 1. Model Deployment and Prediction System
- Input candidate exoplanet data manually or upload a CSV file
- Get predictions from trained machine learning models (Random Forest and XGBoost)
- View prediction confidence and interpretation

### 2. Real-time Analysis System
- Simulate real-time data analysis as new observations come in
- Track trends and patterns in the data over time
- Detect anomalies in the incoming data

### 3. Interactive Analysis Dashboard
- Explore the dataset with interactive visualizations
- Filter and sort data based on various criteria
- Discover relationships between different features

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. First, train and save the models:
   ```
   python train_save_models.py
   ```

2. Launch the Streamlit application:
   ```
   cd streamlit_app
   streamlit run app.py
   ```

3. Open your web browser and navigate to the URL displayed in the terminal (usually http://localhost:8501)

## Data

This project uses the following datasets:
- `cumulative_2024.10.04_10.09.03.csv`: Kepler Objects of Interest (KOI) data
- `kepler_stellar_data.csv`: Stellar properties for Kepler targets

## Technologies Used

- Python
- Streamlit
- HoloViews
- Pandas
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

## Project Structure

```
├── models/                  # Directory for saved models
├── streamlit_app/           # Streamlit application
│   ├── app.py               # Main application file
│   ├── model_prediction.py  # Model prediction page
│   ├── realtime_analysis.py # Real-time analysis page
│   └── interactive_dashboard.py # Interactive dashboard page
├── train_save_models.py     # Script to train and save models
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```
