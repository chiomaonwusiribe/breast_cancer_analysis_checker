from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_and_split_breast_cancer():
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test, feature_names

def train_logistic_regression_bc(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def train_svm_bc(X_train, y_train, X_test, y_test):
    model = SVC(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def train_decision_tree_bc(X_train, y_train, X_test, y_test, random_state=42):
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def train_random_forest_bc(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_and_split_breast_cancer()
    
    lr_accuracy = train_logistic_regression_bc(X_train, y_train, X_test, y_test)
    svm_accuracy = train_svm_bc(X_train, y_train, X_test, y_test)
    dt_accuracy_42 = train_decision_tree_bc(X_train, y_train, X_test, y_test, random_state=42)
    dt_accuracy_none = train_decision_tree_bc(X_train, y_train, X_test, y_test, random_state=None)
    rf_accuracy = train_random_forest_bc(X_train, y_train, X_test, y_test)
    
    print("Breast Cancer Dataset - Model Performance (Accuracy):")
    print(f"Logistic Regression: {lr_accuracy:.4f}")
    print(f"SVM: {svm_accuracy:.4f}")
    print(f"Decision Tree (random_state=42): {dt_accuracy_42:.4f}")
    print(f"Decision Tree (random_state=None): {dt_accuracy_none:.4f}")
    print(f"Random Forest: {rf_accuracy:.4f}")