import streamlit as st
import pandas as pd
import joblib
import yaml
import sys
import os

# Set page config
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)


# Load config and model
@st.cache_resource
def load_model():
    with open('config.yaml') as f:
        config = yaml.safe_load(f)

    model_path = os.path.join(config['paths']['models'], 'random_forest_model.pkl')
    return joblib.load(model_path)


# Title
st.title("🚢 Titanic Survival Prediction")
st.markdown("Predict whether a passenger would have survived the Titanic disaster")

# Sidebar for input
st.sidebar.header("Passenger Details")

# Input fields
pclass = st.sidebar.selectbox("Passenger Class", [1, 2, 3])
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
age = st.sidebar.slider("Age", 0, 100, 30)
sibsp = st.sidebar.slider("Siblings/Spouses", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children", 0, 6, 0)
fare = st.sidebar.slider("Fare", 0, 500, 50)
embarked = st.sidebar.selectbox("Embarked", ["Cherbourg", "Queenstown", "Southampton"])


# Preprocess inputs
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked):
    df = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [1 if sex == 'Male' else 0],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'FamilySize': [sibsp + parch + 1],
        'IsAlone': [1 if (sibsp + parch) == 0 else 0]
    })

    # One-hot encode embarked
    embarked_mapping = {'Cherbourg': 'C', 'Queenstown': 'Q', 'Southampton': 'S'}
    embarked_code = embarked_mapping[embarked]

    for port in ['Q', 'S']:  # Drop first: 'C' is reference
        df[f'Embarked_{port}'] = [1 if embarked_code == port else 0]

    return df


# Predict button
if st.sidebar.button("Predict Survival", type="primary"):
    # Load model
    model = load_model()

    # Preprocess input
    input_df = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked)

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Display result
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prediction Result")
        if prediction == 1:
            st.success("✅ **Survived**", icon="✅")
        else:
            st.error("❌ **Did Not Survive**", icon="❌")

        st.metric("Survival Probability", f"{probability:.1%}")

    with col2:
        st.subheader("Feature Importance")
        # Show explanation
        st.write("Key factors influencing survival:")
        st.markdown("""
        - **Class**: 1st class passengers had higher survival rates
        - **Sex**: Women and children were prioritized
        - **Age**: Younger passengers had better chances
        - **Family**: Traveling with family increased odds
        """)

# Main area - show some analysis
st.header("📊 Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Survival by Class")
    # You could load and show actual data visualizations
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/6d/Titanic-survivors-by-class.png",
             caption="Survival rates by passenger class")

with col2:
    st.subheader("Survival by Sex")
    st.image("https://upload.wikimedia.org/wikipedia/commons/7/71/Titanic-survivors-by-sex.png",
             caption="Survival rates by gender")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Model: Random Forest Classifier")