"""
Main training script for Titanic prediction model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import yaml
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor


def train_model():
    """Main training pipeline"""
    # Load config
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    # Initialize preprocessor
    preprocessor = DataPreprocessor()

    # Load and preprocess data
    print("Loading data...")
    df = preprocessor.load_data(config['paths']['raw_data'])
    df_processed = preprocessor.preprocess(df)

    # Prepare features and target
    X = df_processed.drop('Survived', axis=1)
    y = df_processed['Survived']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['preprocessing']['test_size'],
        random_state=config['preprocessing']['random_state']
    )

    # Train model
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=config['model']['random_forest']['n_estimators'],
        max_depth=config['model']['random_forest']['max_depth'],
        random_state=config['model']['random_forest']['random_state']
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    model_path = os.path.join(config['paths']['models'], 'random_forest_model.pkl')
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 5 Features:")
    print(feature_importance.head())

    return model, accuracy


if __name__ == "__main__":
    train_model()