from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_svm_wine():
    # Load wine dataset
    data = load_wine()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    # Split data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)
    
    # Train model
    model = SVC(random_state=50)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = SVC(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy, feature_names

if __name__ == "__main__":
    accuracy, _ = train_svm_wine()
    print(f"SVM Accuracy on Wine Dataset: {accuracy:.4f}")