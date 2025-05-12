import numpy as numpy
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_and_split_data(test_size=0.3, random_state=42):
    data = load_breast_cancer()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, data.feature_names

def train_logistic_regression(X_train, y_train, X_test):
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def train_svm(X_train, y_train, X_test):
    model = SVC(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def train_decision_tree(X_train, y_train, X_test, random_state=42):
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def train_random_forest(X_train, y_train, X_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Load data
X_train, X_test, y_train, y_test, feature_names = load_and_split_data()

# Train and evaluate models
lr_accuracy = train_logistic_regression(X_train, y_train, X_test)
svm_accuracy = train_svm(X_train, y_train, X_test)
dt_accuracy_42 = train_decision_tree(X_train, y_train, X_test, random_state=42)
dt_accuracy_none = train_decision_tree(X_train, y_train, X_test, random_state=None)
rf_accuracy = train_random_forest(X_train, y_train, X_test)

# Store results
results = {
    'Logistic Regression': lr_accuracy,
    'SVM': svm_accuracy,
    'Decision Tree (random_state=42)': dt_accuracy_42,
    'Decision Tree (random_state=None)': dt_accuracy_none,
    'Random Forest': rf_accuracy
}

if __name__ == "__main__":
    print("Model Performance (Accuracy):")
    for model, accuracy in results.items():
        print(f"{model}: {accuracy:.4f}")