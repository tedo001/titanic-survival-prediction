"""
SIMPLE Titanic model training - Guaranteed to work on Windows
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("=" * 60)
print("SIMPLE TITANIC MODEL TRAINER")
print("=" * 60)

# 1. Create folders
print("\n📁 Creating folders...")
folders = ['data', 'data/raw', 'data/processed', 'models', 'reports']
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"  Created: {folder}")

# 2. Create synthetic data (no internet needed)
print("\n📊 Creating synthetic Titanic data...")

np.random.seed(42)
n_samples = 891

data = {
    'PassengerId': range(1, n_samples + 1),
    'Survived': np.random.choice([0, 1], n_samples, p=[0.62, 0.38]),
    'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55]),
    'Sex': np.random.choice([0, 1], n_samples, p=[0.35, 0.65]),  # 0=Female, 1=Male
    'Age': np.random.normal(29, 14, n_samples).clip(0, 80),
    'SibSp': np.random.poisson(0.5, n_samples).clip(0, 8),
    'Parch': np.random.poisson(0.4, n_samples).clip(0, 6),
    'Fare': np.random.exponential(32, n_samples).clip(0, 512),
    'Embarked_Q': np.random.choice([0, 1], n_samples, p=[0.91, 0.09]),
    'Embarked_S': np.random.choice([0, 1], n_samples, p=[0.28, 0.72])
}

# Add missing values to Age
mask = np.random.random(n_samples) < 0.2
data['Age'] = np.where(mask, np.nan, data['Age'])

df = pd.DataFrame(data)

# Fill missing ages with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Add engineered features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

print(f"✅ Created synthetic dataset with {len(df)} passengers")
print(f"  Survived: {df['Survived'].sum()} ({df['Survived'].mean():.1%})")
print(f"  Columns: {list(df.columns)}")

# 3. Prepare data for training
print("\n🔧 Preparing data for training...")

# Select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'FamilySize', 'IsAlone', 'Embarked_Q', 'Embarked_S']
X = df[features]
y = df['Survived']

print(f"  Features: {len(features)}")
print(f"  Target: Survival (0=No, 1=Yes)")

# 4. Split data
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Training samples: {len(X_train)}")
print(f"  Test samples: {len(X_test)}")

# 5. Train model
print("\n🤖 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model.fit(X_train, y_train)

# 6. Evaluate
print("\n📈 Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"  Test Accuracy: {accuracy:.2%}")

# 7. Save model
model_path = 'models/random_forest_model.pkl'
joblib.dump(model, model_path)
print(f"\n💾 Model saved to: {model_path}")

# Save feature names
feature_names = list(X.columns)
joblib.dump(feature_names, 'models/feature_names.pkl')
print(f"  Feature names saved")

# 8. Save data
df.to_csv('data/raw/titanic.csv', index=False)
print(f"  Data saved to: data/raw/titanic.csv")

# 9. Create sample predictions
print("\n🎯 Sample Predictions:")
print("-" * 40)

# Create some example passengers
examples = [
    {"Type": "Woman, 1st Class", "Pclass": 1, "Sex": 0, "Age": 25, "Fare": 200},
    {"Type": "Man, 3rd Class", "Pclass": 3, "Sex": 1, "Age": 40, "Fare": 20},
    {"Type": "Child, 2nd Class", "Pclass": 2, "Sex": 0, "Age": 10, "Fare": 50},
]

for example in examples:
    # Create full feature set
    input_data = {
        'Pclass': example['Pclass'],
        'Sex': example['Sex'],
        'Age': example['Age'],
        'SibSp': 0,
        'Parch': 0,
        'Fare': example['Fare'],
        'FamilySize': 1,
        'IsAlone': 1,
        'Embarked_Q': 0,
        'Embarked_S': 1
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])[features]

    # Predict
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    survival = "SURVIVED" if pred == 1 else "DIED"
    print(f"  {example['Type']}: {survival} ({proba:.1%} chance)")

print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE! Ready to run the Streamlit app.")
print("=" * 60)
print("\nNext step: Run this command:")
print("  streamlit run simple_app.py")