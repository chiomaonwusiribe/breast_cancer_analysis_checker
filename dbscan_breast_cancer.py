from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

def train_dbscan_breast_cancer(eps=3.0, min_samples=5):
    # Load breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    feature_names = data.feature_names
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    
    # Evaluate (only if at least 2 clusters are formed)
    silhouette = -1.0  # Default for invalid cases
    if len(np.unique(labels[labels != -1])) >= 2:
        silhouette = silhouette_score(X_scaled[labels != -1], labels[labels != -1])
    
    return silhouette, feature_names

if __name__ == "__main__":
    silhouette, _ = train_dbscan_breast_cancer()
    print(f"DBSCAN Silhouette Score (eps=3.0, min_samples=5): {silhouette:.4f}")