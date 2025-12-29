"""
Script to train and save the Titanic prediction model
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data/raw', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)


def download_and_prepare_data():
    """Download Titanic dataset from Kaggle or use local file"""
    try:
        # If you have the data locally
        df = pd.read_csv('data/raw/titanic.csv')
        print("✅ Data loaded from local file")
    except:
        print("Downloading Titanic dataset...")
        # Load from seaborn (requires internet)
        try:
            import seaborn as sns
            df = sns.load_dataset('titanic')
            df.to_csv('data/raw/titanic.csv', index=False)
            print("✅ Data downloaded and saved")
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            print("Creating synthetic data as fallback...")
            df = create_synthetic_data()

    return df


def create_synthetic_data():
    """Create synthetic Titanic-like data as fallback"""
    np.random.seed(42)
    n_passengers = 891

    # Create synthetic data
    data = {
        'survived': np.random.choice([0, 1], n_passengers, p=[0.62, 0.38]),
        'pclass': np.random.choice([1, 2, 3], n_passengers, p=[0.24, 0.21, 0.55]),
        'sex': np.random.choice(['male', 'female'], n_passengers, p=[0.65, 0.35]),
        'age': np.random.normal(29, 14, n_passengers).clip(0, 80),
        'sibsp': np.random.poisson(0.5, n_passengers).clip(0, 8),
        'parch': np.random.poisson(0.4, n_passengers).clip(0, 6),
        'fare': np.random.exponential(32, n_passengers).clip(0, 512),
        'embarked': np.random.choice(['S', 'C', 'Q'], n_passengers, p=[0.72, 0.19, 0.09])
    }

    # Add some missing values
    mask = np.random.random(n_passengers) < 0.2
    data['age'] = np.where(mask, np.nan, data['age'])
    mask = np.random.random(n_passengers) < 0.1
    data['embarked'] = np.where(mask, np.nan, data['embarked'])

    df = pd.DataFrame(data)
    df.to_csv('data/raw/titanic.csv', index=False)
    print("✅ Synthetic dataset created")

    return df


def preprocess_data(df):
    """Preprocess the Titanic dataset"""
    # Select relevant columns
    df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']].copy()

    # Rename columns for consistency
    df.columns = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Encode categorical variables
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])  # Male=1, Female=0

    # One-hot encode embarked
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)
    df = pd.concat([df, embarked_dummies], axis=1)
    df = df.drop('Embarked', axis=1)

    return df


def train_and_save_model():
    """Main training function"""
    print("\n" + "=" * 50)
    print("TITANIC SURVIVAL PREDICTION MODEL TRAINING")
    print("=" * 50)

    print("\nStep 1: Loading data...")
    df = download_and_prepare_data()
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\nStep 2: Preprocessing data...")
    df_processed = preprocess_data(df)
    print(f"Processed data shape: {df_processed.shape}")
    print(f"Missing values after preprocessing:\n{df_processed.isnull().sum()}")

    print("\nStep 3: Preparing features and target...")
    X = df_processed.drop('Survived', axis=1)
    y = df_processed['Survived']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")

    print("\nStep 4: Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    print("\nStep 5: Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n✅ Model trained successfully!")
    print(f"Test Accuracy: {accuracy:.4f}")

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    model_path = 'models/random_forest_model.pkl'
    joblib.dump(model, model_path)
    print(f"\n📁 Model saved to: {model_path}")

    # Save feature names for reference in app
    feature_names = list(X.columns)
    joblib.dump(feature_names, 'models/feature_names.pkl')
    print(f"Feature names saved: {len(feature_names)} features")

    # Create visualization of feature importance
    print("\n📈 Creating feature importance visualization...")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n🔝 Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()

    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/feature_importance.png', bbox_inches='tight', dpi=100)
    print("📊 Feature importance plot saved to: reports/feature_importance.png")

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('reports/confusion_matrix.png', bbox_inches='tight', dpi=100)
    print("📊 Confusion matrix saved to: reports/confusion_matrix.png")

    # Save processed data
    df_processed.to_csv('data/processed/titanic_processed.csv', index=False)
    print("💾 Processed data saved to: data/processed/titanic_processed.csv")

    # Print sample predictions
    print("\n🎯 Sample Predictions:")
    sample_data = X_test.head(3).copy()
    sample_predictions = model.predict(sample_data)
    sample_probabilities = model.predict_proba(sample_data)[:, 1]

    for i in range(len(sample_data)):
        actual = y_test.iloc[i] if i < len(y_test) else "N/A"
        print(f"  Sample {i + 1}: Predicted={sample_predictions[i]} "
              f"(Prob={sample_probabilities[i]:.2f}), Actual={actual}")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE! ✅")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run the Streamlit app: streamlit run app.py")
    print("2. Open browser to http://localhost:8501")
    print("3. Test predictions with different passenger profiles")

    return model, accuracy


if __name__ == "__main__":
    train_and_save_model()