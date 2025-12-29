"""
TRAIN TITANIC MODEL - Run this first!
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("STARTING TITANIC MODEL TRAINING")
print("=" * 50)

# 1. Create necessary folders
folders = ['models', 'data', 'data/raw', 'data/processed']
for folder in folders:
    os.makedirs(folder, exist_ok=True)
    print(f"Created: {folder}")

# 2. Create synthetic data (no internet needed!)
print("\nCreating Titanic dataset...")

np.random.seed(42)
n_passengers = 1000

# Create realistic Titanic data
data = {
    'Pclass': np.random.choice([1, 2, 3], n_passengers, p=[0.24, 0.21, 0.55]),
    'Sex': np.random.choice([0, 1], n_passengers, p=[0.35, 0.65]),  # 0=Female, 1=Male
    'Age': np.random.normal(29, 14, n_passengers).clip(0, 80),
    'SibSp': np.random.poisson(0.5, n_passengers).clip(0, 8),
    'Parch': np.random.poisson(0.4, n_passengers).clip(0, 6),
    'Fare': np.random.exponential(32, n_passengers).clip(0, 512).round(2),
    'Embarked_Q': np.random.choice([0, 1], n_passengers, p=[0.91, 0.09]),
    'Embarked_S': np.random.choice([0, 1], n_passengers, p=[0.28, 0.72])
}

# Create target (Survived) based on realistic patterns
survival_prob = []
for i in range(n_passengers):
    prob = 0.5

    # Women had higher survival
    if data['Sex'][i] == 0:  # Female
        prob += 0.3

    # 1st class had higher survival
    if data['Pclass'][i] == 1:
        prob += 0.2
    elif data['Pclass'][i] == 3:
        prob -= 0.2

    # Children had higher survival
    if data['Age'][i] < 18:
        prob += 0.15

    # Higher fare = better survival
    if data['Fare'][i] > 100:
        prob += 0.1

    survival_prob.append(min(0.95, max(0.05, prob)))

# Generate survival outcomes
data['Survived'] = [1 if np.random.random() < p else 0 for p in survival_prob]

# Create DataFrame
df = pd.DataFrame(data)

# Add missing values (like real Titanic data)
missing_mask = np.random.random(n_passengers) < 0.2
df.loc[missing_mask, 'Age'] = np.nan

# Fill missing ages
df['Age'] = df['Age'].fillna(df['Age'].median())

# Add engineered features
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

print(f"Created dataset with {len(df)} passengers")
print(f"   Survived: {df['Survived'].sum()} ({df['Survived'].mean():.1%})")
print(f"   Features: {list(df.columns)}")

# 3. Prepare for training
print("\n Preparing data...")

# Features to use
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
            'FamilySize', 'IsAlone', 'Embarked_Q', 'Embarked_S']

X = df[features]
y = df['Survived']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# 4. Train model
print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nMODEL TRAINED SUCCESSFULLY!")
print(f"   Test Accuracy: {accuracy:.2%}")

# 6. Save model
model_path = 'models/titanic_model.pkl'
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")

# Save feature names
joblib.dump(features, 'models/feature_names.pkl')
print(f"   Feature names saved")

# 7. Save data
df.to_csv('data/raw/titanic.csv', index=False)
print(f"   Data saved to: data/raw/titanic.csv")

# 8. Show feature importance
print("\n🔝 Most Important Features:")
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance.head(5).iterrows():
    print(f"   {row['Feature']}: {row['Importance']:.3f}")

# 9. Show example predictions
print("\nExample Predictions:")
print("-" * 40)

examples = [
    {"Desc": "Woman, 25, 1st Class", "Sex": 0, "Pclass": 1, "Age": 25, "Fare": 200},
    {"Desc": "Man, 40, 3rd Class", "Sex": 1, "Pclass": 3, "Age": 40, "Fare": 20},
    {"Desc": "Girl, 8, 2nd Class", "Sex": 0, "Pclass": 2, "Age": 8, "Fare": 50},
]

for ex in examples:
    input_data = {
        'Pclass': ex['Pclass'],
        'Sex': ex['Sex'],
        'Age': ex['Age'],
        'SibSp': 0,
        'Parch': 0,
        'Fare': ex['Fare'],
        'FamilySize': 1,
        'IsAlone': 1,
        'Embarked_Q': 0,
        'Embarked_S': 1
    }

    input_df = pd.DataFrame([input_data])[features]
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    result = "SURVIVES" if pred == 1 else "PERISHES"
    print(f"   {ex['Desc']}: {result} ({proba:.1%} chance)")

print("\n" + "=" * 50)
print("TRAINING COMPLETE!")
print("=" * 50)
print("\nNext: Run this command:")
print("   streamlit run app.py")
print("\nThe app will open in your browser!")