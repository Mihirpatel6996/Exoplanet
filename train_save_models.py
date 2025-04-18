import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load data
print("Loading data...")
df_cumulative = pd.read_csv('cumulative_2024.10.04_10.09.03.csv')
df_stellar = pd.read_csv('kepler_stellar_data.csv')

# Merge datasets
df_merged = pd.merge(df_cumulative, df_stellar, on='kepid', how='inner')

# Select the relevant columns
selected_columns = ["kepid", "koi_disposition", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", 
                    "koi_fpflag_ec", "koi_period", "koi_prad", "koi_sma", "koi_teq", "koi_insol", 
                    "koi_steff", "koi_srad", "koi_smass", "teff", "logg", "feh", "mass", "radius", 
                    "dens", "kepmag"]
df_selected = df_merged[selected_columns].copy()

# Handle missing values
for col in df_selected.select_dtypes(include=['number']).columns:
    df_selected[col] = df_selected[col].fillna(df_selected[col].median())

for col in df_selected.select_dtypes(include=['object']).columns:
    df_selected[col] = df_selected[col].fillna('unknown')

# Encode categorical features
df_selected = pd.get_dummies(df_selected, columns=['koi_disposition'], prefix=['koi_disposition'])

# Feature engineering
df_selected['koi_prad_radius_ratio'] = df_selected['koi_prad'] / df_selected['radius']
df_selected['koi_sma_radius_ratio'] = df_selected['koi_sma'] / df_selected['radius']
df_selected['koi_insol_teq_product'] = df_selected['koi_insol'] * df_selected['koi_teq']

# Separate features (X) and target (y)
X = df_selected.drop(['koi_disposition_CANDIDATE', 'koi_disposition_CONFIRMED', 'koi_disposition_FALSE POSITIVE'], axis=1)
y = df_selected[['koi_disposition_CANDIDATE', 'koi_disposition_CONFIRMED', 'koi_disposition_FALSE POSITIVE']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save feature names and scaler
print("Saving feature names and scaler...")
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'models/feature_names.joblib')
joblib.dump(scaler, 'models/scaler.joblib')

# Train and save Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=1,
    criterion='gini',
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# Evaluate Random Forest model
rf_predictions = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions, average='weighted')
rf_recall = recall_score(y_test, rf_predictions, average='weighted')
rf_f1 = f1_score(y_test, rf_predictions, average='weighted')

print(f"Random Forest Model Performance:")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")
print(f"F1 Score: {rf_f1:.4f}")

# Save Random Forest model
print("Saving Random Forest model...")
joblib.dump(rf_model, 'models/random_forest_model.joblib')

# Train and save XGBoost model
print("Training XGBoost model...")
xgb_model = XGBClassifier(
    learning_rate=0.1,
    n_estimators=100,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train_scaled, y_train)

# Evaluate XGBoost model
xgb_predictions = xgb_model.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_precision = precision_score(y_test, xgb_predictions, average='weighted')
xgb_recall = recall_score(y_test, xgb_predictions, average='weighted')
xgb_f1 = f1_score(y_test, xgb_predictions, average='weighted')

print(f"\nXGBoost Model Performance:")
print(f"Accuracy: {xgb_accuracy:.4f}")
print(f"Precision: {xgb_precision:.4f}")
print(f"Recall: {xgb_recall:.4f}")
print(f"F1 Score: {xgb_f1:.4f}")

# Save XGBoost model
print("Saving XGBoost model...")
joblib.dump(xgb_model, 'models/xgboost_model.joblib')

print("Models trained and saved successfully!")
