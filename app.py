import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Title
st.title("🚢 Titanic Survival Prediction")
st.markdown("Predict whether a passenger would have survived the Titanic disaster")

# Sidebar for model info
with st.sidebar:
    st.header("Model Information")

    if os.path.exists('models/random_forest_model.pkl'):
        st.success("✅ Model is loaded")
        if os.path.exists('reports/feature_importance.png'):
            st.image('reports/feature_importance.png', caption='Feature Importance')
    else:
        st.warning("⚠️ Model not found")
        st.info("Run `python train_model.py` first to train the model")


# Check if model exists
def load_model():
    model_path = 'models/random_forest_model.pkl'

    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.error("Model file not found! Please train the model first.")
        st.stop()


# Load model
try:
    model = load_model()
    # Load feature names if available
    if os.path.exists('models/feature_names.pkl'):
        feature_names = joblib.load('models/feature_names.pkl')
    else:
        feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                         'FamilySize', 'IsAlone', 'Embarked_Q', 'Embarked_S']
except:
    model = None
    feature_names = []

# Input form
st.header("📝 Passenger Information")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3],
        help="1st class had the highest survival rate"
    )

    sex = st.selectbox(
        "Gender",
        ["Female", "Male"],
        help="Women had priority in lifeboats"
    )

    age = st.slider(
        "Age",
        0, 100, 30,
        help="Children and elderly had different survival rates"
    )

with col2:
    sibsp = st.number_input(
        "Number of Siblings/Spouses",
        0, 8, 0,
        help="Siblings or spouses aboard"
    )

    parch = st.number_input(
        "Number of Parents/Children",
        0, 6, 0,
        help="Parents or children aboard"
    )

    fare = st.slider(
        "Ticket Fare (£)",
        0, 500, 50,
        help="Higher fare often meant better class/location"
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["Southampton", "Cherbourg", "Queenstown"],
        help="S = Southampton, C = Cherbourg, Q = Queenstown"
    )

# Predict button
if st.button("🚀 Predict Survival", type="primary"):
    if model is None:
        st.error("Model not available. Please train the model first.")
    else:
        # Preprocess input
        sex_encoded = 1 if sex == "Male" else 0
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0

        # Prepare input data
        input_data = {
            'Pclass': pclass,
            'Sex': sex_encoded,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'FamilySize': family_size,
            'IsAlone': is_alone,
            'Embarked_Q': 1 if embarked == "Queenstown" else 0,
            'Embarked_S': 1 if embarked == "Southampton" else 0
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure all features are present and in correct order
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0

        input_df = input_df[feature_names]

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Display results
        st.header("📊 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.success(f"## ✅ SURVIVED")
                st.metric("Survival Probability", f"{probability:.1%}")
            else:
                st.error(f"## ❌ DID NOT SURVIVE")
                st.metric("Survival Probability", f"{probability:.1%}")

        with col2:
            # Show influencing factors
            st.subheader("Key Factors")

            factors = []
            if sex == "Female":
                factors.append("✅ Female (higher survival rate)")
            else:
                factors.append("❌ Male (lower survival rate)")

            if pclass == 1:
                factors.append("✅ 1st Class (highest priority)")
            elif pclass == 2:
                factors.append("⚠️ 2nd Class (moderate chances)")
            else:
                factors.append("❌ 3rd Class (lowest priority)")

            if age < 18:
                factors.append("✅ Child ('women and children first')")
            elif age > 50:
                factors.append("⚠️ Elderly (lower mobility)")

            if is_alone:
                factors.append("⚠️ Traveling alone")
            else:
                factors.append("✅ With family")

            for factor in factors:
                st.write(factor)

# Historical facts
st.markdown("---")
st.header("📜 Historical Facts")

facts = [
    "**Overall Survival Rate**: 31.9% (710 of 2,224 passengers)",
    "**Women's Survival**: 74.2% of women survived vs 18.9% of men",
    "**Class Matters**: 62.9% of 1st class survived vs 25.2% of 3rd class",
    "**Children First**: 52.3% of children survived",
    "**The Captain**: Captain Smith went down with the ship"
]

for fact in facts:
    st.write(f"• {fact}")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit | Random Forest Classifier</p>
        <p><small>Note: This is a machine learning prediction based on historical patterns.</small></p>
    </div>
    """,
    unsafe_allow_html=True
)