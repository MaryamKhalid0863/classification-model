# Iris Dataset Classification
# This script demonstrates a simple machine learning workflow using the Iris dataset.

# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

def load_data():
    """Load the Iris dataset and return as a pandas DataFrame."""
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['target'] = iris.target
    return data

def split_data(data, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    X = data.drop('target', axis=1)
    y = data['target']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return the accuracy."""
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def main():
    # Load Data
    data = load_data()
    
    # Split Data
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Train Model
    model = train_model(X_train, y_train)
    
    # Evaluate Model
    accuracy = evaluate_model(model, X_test, y_test)
    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    main()
