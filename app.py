"""
SIMPLE Titanic Survival Predictor - Streamlit App
"""
import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered"
)

# Title
st.title("🚢 Titanic Survival Prediction")
st.markdown("### Will You Survive the Titanic Disaster?")


# Check if model exists
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/random_forest_model.pkl')
        return model
    except:
        st.error("⚠️ Model not found. Please run `python simple_train.py` first.")
        return None


@st.cache_resource
def load_features():
    try:
        features = joblib.load('models/feature_names.pkl')
        return features
    except:
        return ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare',
                'FamilySize', 'IsAlone', 'Embarked_Q', 'Embarked_S']


# Load model and features
model = load_model()
features = load_features()

# Sidebar with instructions
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. Fill in passenger details
    2. Click **Predict Survival**
    3. See your predicted fate!

    **Historical Facts:**
    - Women had 74% survival rate
    - 1st class had 63% survival rate
    - Children had priority in lifeboats
    """)

    if model:
        st.success("✅ Model loaded")
    else:
        st.warning("❌ Model not found")

# Main input form
st.header("📝 Passenger Details")

# Create columns for better layout
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3],
        help="1st class had the best survival chances"
    )

    sex = st.selectbox(
        "Gender",
        ["Female", "Male"],
        index=0
    )

    age = st.slider(
        "Age",
        0, 100, 30
    )

with col2:
    sibsp = st.number_input(
        "Siblings/Spouses",
        0, 8, 0,
        help="Traveling with family?"
    )

    parch = st.number_input(
        "Parents/Children",
        0, 6, 0
    )

    fare = st.slider(
        "Ticket Fare (£)",
        0, 500, 50,
        help="Higher fare = better location on ship"
    )

# Embarkation port
embarked = st.selectbox(
    "Embarked From",
    ["Southampton", "Cherbourg", "Queenstown"]
)

# Predict button
if st.button("🔮 Predict My Fate", type="primary", use_container_width=True):
    if model is None:
        st.error("Please train the model first (run simple_train.py)")
    else:
        # Preprocess input
        sex_code = 0 if sex == "Female" else 1
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0

        # Prepare input data
        input_data = {
            'Pclass': pclass,
            'Sex': sex_code,
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

        # Make sure all features are present
        for feature in features:
            if feature not in input_df.columns:
                input_df[feature] = 0

        # Reorder columns to match training
        input_df = input_df[features]

        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        # Display result
        st.header("🎯 Prediction Result")

        if prediction == 1:
            st.success(f"""
            ## ✅ YOU SURVIVE!

            **Survival Probability:** {probability:.1%}

            You would have been one of the lucky ones to make it onto a lifeboat.
            """)
        else:
            st.error(f"""
            ## ❌ YOU PERISH

            **Survival Probability:** {probability:.1%}

            Sadly, you would not have made it onto a lifeboat in time.
            """)

        # Show influencing factors
        st.subheader("📊 What Affected Your Chances:")

        factors = []

        # Gender factor
        if sex == "Female":
            factors.append("✅ **Female** - Women had priority in lifeboats")
        else:
            factors.append("❌ **Male** - 'Women and children first' policy")

        # Class factor
        if pclass == 1:
            factors.append("✅ **First Class** - Closest to lifeboats")
        elif pclass == 2:
            factors.append("⚠️ **Second Class** - Moderate chances")
        else:
            factors.append("❌ **Third Class** - Far from lifeboat decks")

        # Age factor
        if age < 18:
            factors.append("✅ **Child** - Priority evacuation")
        elif age > 50:
            factors.append("⚠️ **Elderly** - Lower mobility")

        # Family factor
        if is_alone:
            factors.append("⚠️ **Traveling Alone** - No family to help")
        else:
            factors.append("✅ **With Family** - Better support system")

        # Fare factor
        if fare > 100:
            factors.append("✅ **High Fare** - Likely better cabin location")

        for factor in factors:
            st.write(factor)

# Historical context
st.markdown("---")
st.header("📜 Historical Context")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Passengers", "2,224")
    st.caption("Aboard the Titanic")

with col2:
    st.metric("Survivors", "710")
    st.caption("31.9% survival rate")

with col3:
    st.metric("Fatalities", "1,514")
    st.caption("68.1% perished")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p><em>Built with Streamlit • Random Forest Classifier • Titanic Dataset</em></p>
        <p><small>This is a predictive model based on historical patterns, not actual fate.</small></p>
    </div>
    """,
    unsafe_allow_html=True
)