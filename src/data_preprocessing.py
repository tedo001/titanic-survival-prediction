import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import yaml


class DataPreprocessor:
    def __init__(self, config_path='config.yaml'):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

    def load_data(self, path):
        """Load and inspect data"""
        df = pd.read_csv(path)
        print(f"Data shape: {df.shape}")
        print(f"Missing values:\n{df.isnull().sum()}")
        return df

    def preprocess(self, df):
        """Full preprocessing pipeline"""
        # Drop columns
        cols_to_drop = self.config['preprocessing']['features_to_drop']
        df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

        # Handle missing values
        df['Age'] = df['Age'].fillna(df['Age'].median())
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

        # Feature engineering
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

        # Encode categorical
        le = LabelEncoder()
        df['Sex'] = le.fit_transform(df['Sex'])
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

        return df


# Unit test in PyCharm
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data('data/raw/titanic.csv')
    df_processed = preprocessor.preprocess(df)
    print(f"Processed shape: {df_processed.shape}")