# src/train.py
"""
Simple ML training script for MLOps lab
Dataset: Iris (sklearn built-in)
Model: Logistic Regression
"""

import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_data():
    """Load dataset"""
    data = load_iris()
    X = data.data
    y = data.target
    return X, y


def train_model(X_train, y_train):
    """Train ML model"""
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model"""
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    return acc


def save_model(model, path="models/model.pkl"):
    """Save trained model"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def main():
    print("🚀 Starting training pipeline...")

    # Load data
    X, y = load_data()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = train_model(X_train, y_train)

    # Evaluate
    accuracy = evaluate_model(model, X_test, y_test)
    print(f"✅ Model Accuracy: {accuracy:.4f}")

    # Save model
    save_model(model)
    print("💾 Model saved to models/model.pkl")


if __name__ == "__main__":
    main()