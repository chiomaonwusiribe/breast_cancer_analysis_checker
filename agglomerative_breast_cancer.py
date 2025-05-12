from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def train_agglomerative_breast_cancer():
    # Load breast cancer dataset
    data = load_breast_cancer()
    X = data.data
    feature_names = data.feature_names
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = AgglomerativeClustering(n_clusters=2)
    labels = model.fit_predict(X_scaled)
    
    # Evaluate
    silhouette = silhouette_score(X_scaled, labels)
    
    return silhouette, feature_names

if __name__ == "__main__":
    silhouette, _ = train_agglomerative_breast_cancer()
    print(f"Agglomerative Clustering Silhouette Score: {silhouette:.4f}")