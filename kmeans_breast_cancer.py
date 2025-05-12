from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def train_kmeans_breast_cancer(random_state=42):
    # Load breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    feature_names = data.feature_names
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = KMeans(n_clusters=2, random_state=random_state)
    labels = model.fit_predict(X_scaled)
    
    # Evaluate
    silhouette = silhouette_score(X_scaled, labels)
    
    return silhouette, feature_names

if __name__ == "__main__":
    silhouette_42, _ = train_kmeans_breast_cancer(random_state=42)
    silhouette_none, _ = train_kmeans_breast_cancer(random_state=None)
    print(f"K-Means Silhouette Score (random_state=42): {silhouette_42:.4f}")
    print(f"K-Means Silhouette Score (random_state=None): {silhouette_none:.4f}")