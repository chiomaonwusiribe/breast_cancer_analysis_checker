from flask import Flask, render_template, request, send_file
from logistic_regression_wine import train_logistic_regression_wine
from svm_wine import train_svm_wine
from decision_tree_wine import train_decision_tree_wine
from breast_cancer_experiment import (
    load_and_split_breast_cancer,
    train_logistic_regression_bc,
    train_svm_bc,
    train_decision_tree_bc,
    train_random_forest_bc
)
from kmeans_breast_cancer import train_kmeans_breast_cancer
from agglomerative_breast_cancer import train_agglomerative_breast_cancer
from dbscan_breast_cancer import train_dbscan_breast_cancer
from gmm_breast_cancer import train_gmm_breast_cancer
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

app = Flask(__name__)

# Load Random Forest model, scaler, and top features
model = joblib.load('models/random_forest_model.joblib')
scaler = joblib.load('models/scaler.joblib')
top_feature_indices = joblib.load('models/top_feature_indices.joblib')
top_feature_names = joblib.load('models/top_feature_names.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    # Wine dataset (classification) results
    wine_results = {
        'Logistic Regression': train_logistic_regression_wine()[0],
        'SVM': train_svm_wine()[0],
        'Decision Tree (random_state=42)': train_decision_tree_wine(random_state=42)[0],
        'Decision Tree (random_state=None)': train_decision_tree_wine(random_state=None)[0]
    }
    wine_feature_names = train_logistic_regression_wine()[1]
    
    # Breast cancer dataset (classification) results
    X_train, X_test, y_train, y_test, bc_feature_names = load_and_split_breast_cancer()
    bc_classification_results = {
        'Logistic Regression': train_logistic_regression_bc(X_train, y_train, X_test, y_test),
        'SVM': train_svm_bc(X_train, y_train, X_test, y_test),
        'Decision Tree (random_state=42)': train_decision_tree_bc(X_train, y_train, X_test, y_test, random_state=42),
        'Decision Tree (random_state=None)': train_decision_tree_bc(X_train, y_train, X_test, y_test, random_state=None),
        'Random Forest': train_random_forest_bc(X_train, y_train, X_test, y_test)
    }
    
    # Breast cancer dataset (clustering) results
    bc_clustering_results = {
        'K-Means (random_state=42)': train_kmeans_breast_cancer(random_state=42)[0],
        'K-Means (random_state=None)': train_kmeans_breast_cancer(random_state=None)[0],
        'Agglomerative Clustering': train_agglomerative_breast_cancer()[0],
        'DBSCAN (eps=3.0, min_samples=5)': train_dbscan_breast_cancer()[0],
        'Gaussian Mixture Model': train_gmm_breast_cancer()[0]
    }
    
    # Prediction handling
    prediction = None
    probability = None
    error = None
    show_observations = request.args.get('show_observations', 'false').lower() == 'true'
    
    if request.method == 'POST':
        try:
            # Get input features (only top features)
            features = [float(request.form.get(feature.replace(' ', '_'))) for feature in top_feature_names]
            # Create full feature array with zeros for non-top features
            full_features = np.zeros(30)  # 30 features in breast cancer dataset
            full_features[top_feature_indices] = features
            full_features = full_features.reshape(1, -1)
            
            # Scale features
            features_scaled = scaler.transform(full_features)
            
            # Predict
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0][prediction] * 100
            prediction = 'Benign' if prediction == 1 else 'Malignant'
            
        except Exception as e:
            error = f"Error: Please ensure all inputs are valid numbers. Details: {str(e)}"
    
    return render_template('index.html', 
                         wine_results=wine_results, 
                         bc_classification_results=bc_classification_results,
                         bc_clustering_results=bc_clustering_results,
                         wine_feature_names=wine_feature_names, 
                         bc_feature_names=bc_feature_names,
                         top_feature_names=top_feature_names,
                         prediction=prediction,
                         probability=probability,
                         error=error,
                         show_observations=show_observations)

@app.route('/download_report')
def download_report():
    return send_file('static/report.pdf', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)