from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os

def train_and_save_model(n_top_features=5):
    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Compute feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:n_top_features]
    top_features = feature_names[top_indices].tolist()
    
    # Save model, scaler, and top features with compression
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest_model.joblib', compress=3)
    joblib.dump(scaler, 'models/scaler.joblib', compress=3)
    joblib.dump(top_indices, 'models/top_feature_indices.joblib', compress=3)
    joblib.dump(top_features, 'models/top_feature_names.joblib', compress=3)
    
    return scaler, model, top_features, top_indices

if __name__ == "__main__":
    scaler, model, top_features, top_indices = train_and_save_model()
    print("Random Forest model, scaler, and top features saved successfully.")
    print("Top 5 features:", top_features)