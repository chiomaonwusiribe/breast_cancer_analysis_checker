from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_logistic_regression_wine():
    # Load wine dataset
    data = load_wine()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
    
    # Train model
    model = LogisticRegression(max_iter=10000, random_state=50)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, feature_names

if __name__ == "__main__":
    accuracy, _ = train_logistic_regression_wine()
    print(f"Logistic Regression Accuracy on Wine Dataset: {accuracy:.4f}")