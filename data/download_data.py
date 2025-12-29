"""
Download Titanic dataset automatically
"""
import pandas as pd
import os
from sklearn.datasets import fetch_openml


def download_titanic_data():
    """Download Titanic dataset from multiple sources"""

    # Create directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("Downloading Titanic dataset...")

    # Try multiple sources
    try:
        # Method 1: From seaborn (easiest)
        import seaborn as sns
        print("Downloading from seaborn...")
        df = sns.load_dataset('titanic')
        df.to_csv('data/raw/titanic.csv', index=False)
        print("✅ Dataset downloaded from seaborn and saved to data/raw/titanic.csv")

    except Exception as e:
        print(f"Seaborn download failed: {e}")

        try:
            # Method 2: From sklearn (fetch_openml)
            print("Trying sklearn openml...")
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context

            titanic = fetch_openml(name='titanic', version=1, as_frame=True)
            df = titanic.frame
            df.to_csv('data/raw/titanic.csv', index=False)
            print("✅ Dataset downloaded from openml and saved to data/raw/titanic.csv")

        except Exception as e2:
            print(f"OpenML download failed: {e2}")

            # Method 3: Create synthetic data as fallback
            print("Creating synthetic dataset as fallback...")
            create_synthetic_data()

    return 'data/raw/titanic.csv'


def create_synthetic_data():
    """Create synthetic Titanic-like data as fallback"""
    import numpy as np

    np.random.seed(42)
    n_passengers = 891

    # Create synthetic data
    data = {
        'PassengerId': range(1, n_passengers + 1),
        'Survived': np.random.choice([0, 1], n_passengers, p=[0.62, 0.38]),
        'Pclass': np.random.choice([1, 2, 3], n_passengers, p=[0.24, 0.21, 0.55]),
        'Name': [f'Passenger {i}' for i in range(1, n_passengers + 1)],
        'Sex': np.random.choice(['male', 'female'], n_passengers, p=[0.65, 0.35]),
        'Age': np.random.normal(29, 14, n_passengers).clip(0, 80),
        'SibSp': np.random.poisson(0.5, n_passengers).clip(0, 8),
        'Parch': np.random.poisson(0.4, n_passengers).clip(0, 6),
        'Ticket': [f'Ticket{i}' for i in range(1, n_passengers + 1)],
        'Fare': np.random.exponential(32, n_passengers).clip(0, 512),
        'Cabin': [f'C{np.random.randint(1, 100)}' if np.random.random() > 0.77 else np.nan for _ in
                  range(n_passengers)],
        'Embarked': np.random.choice(['S', 'C', 'Q'], n_passengers, p=[0.72, 0.19, 0.09])
    }

    # Add some missing values
    mask = np.random.random(n_passengers) < 0.2
    data['Age'] = np.where(mask, np.nan, data['Age'])

    df = pd.DataFrame(data)
    df.to_csv('data/raw/titanic.csv', index=False)
    print("✅ Synthetic dataset created and saved to data/raw/titanic.csv")

    return df


if __name__ == "__main__":
    download_titanic_data()
    print("\nDataset ready! You can now run train_model.py")