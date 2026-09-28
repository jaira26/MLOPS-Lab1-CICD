import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def train_model():
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    print("Model trained successfully!")

if __name__ == "__main__":
    train_model()