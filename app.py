"""
TITANIC SURVIVAL PREDICTOR - Streamlit App
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
        font-size: 1.8rem;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .survived {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        color: #155724;
        border: 3px solid #28a745;
    }
    .not-survived {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        color: #721c24;
        border: 3px solid #dc3545;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        font-size: 1.2rem;
        height: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ========== TITLE ==========
st.markdown('<h1 class="main-header">🚢 Titanic Survival Prediction</h1>', unsafe_allow_html=True)
st.markdown("### Predict your fate on the RMS Titanic's maiden voyage")

# ========== SIDEBAR ==========
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/600px-RMS_Titanic_3.jpg",
             caption="RMS Titanic - The 'Unsinkable' Ship")

    st.header("📋 Instructions")
    st.markdown("""
    1. Fill in passenger details
    2. Click **Predict Survival** 
    3. See your predicted fate!
    """)

    st.header("📊 Historical Facts")
    st.markdown("""
    - **Women**: 74% survival rate
    - **1st Class**: 63% survival rate  
    - **Children**: Priority in lifeboats
    - **Overall**: Only 32% survived
    """)

    # Check if model exists
    st.header("🔧 System Status")
    if os.path.exists('models/titanic_model.pkl'):
        st.success("✅ Model loaded and ready!")
        st.metric("Model Status", "Active")
    else:
        st.error("❌ Model not found!")
        st.markdown("""
        **To fix this:**

        1. Run `python train_model.py`
        2. Wait for training to complete
        3. Refresh this page

        This will create and save the model.
        """)


# ========== LOAD MODEL ==========
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model = joblib.load('models/titanic_model.pkl')
        features = joblib.load('models/feature_names.pkl')
        return model, features
    except:
        return None, None


model, features = load_model()

# ========== MAIN INPUT FORM ==========
st.header("📝 Passenger Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Personal Details")

    sex = st.radio(
        "Gender",
        ["👩 Female", "👨 Male"],
        horizontal=True,
        help="Women had priority in lifeboats"
    )

    age = st.slider(
        "Age",
        0, 100, 30,
        help="Children (<18) were prioritized"
    )

    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3],
        format_func=lambda x: f"{x}{'st' if x == 1 else 'nd' if x == 2 else 'rd'} Class",
        help="Higher class = better location on ship"
    )

with col2:
    st.subheader("Travel Details")

    sibsp = st.number_input(
        "Number of Siblings/Spouses",
        0, 8, 0,
        help="Traveling with family?"
    )

    parch = st.number_input(
        "Number of Parents/Children",
        0, 6, 0,
        help="Parents or children aboard"
    )

    fare = st.slider(
        "Ticket Fare (£)",
        0, 500, 50,
        10,
        help="Higher fare typically meant better accommodations"
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        ["Southampton 🏴󠁧󠁢󠁥󠁮󠁧󠁿", "Cherbourg 🇫🇷", "Queenstown 🇮🇪"],
        help="Where the passenger boarded"
    )

# ========== PREDICTION BUTTON ==========
predict_button = st.button("🔮 PREDICT MY FATE", type="primary", use_container_width=True)

if predict_button:
    if model is None:
        st.error("""
        ## ⚠️ Model Not Found!

        Please run the training script first:

        ```bash
        python train_model.py
        ```

        This will create and save the machine learning model.

        After training completes, refresh this page.
        """)

        # Show temporary demo prediction
        st.warning("Showing demo prediction (not using real model):")

        # Simple rule-based prediction for demo
        score = 0
        if "Female" in sex:
            score += 3
        if pclass == 1:
            score += 2
        elif pclass == 2:
            score += 1
        if age < 18:
            score += 2
        if fare > 100:
            score += 1

        probability = min(0.95, score / 8)
        prediction = 1 if probability > 0.5 else 0

    else:
        # ========== PREPARE INPUT DATA ==========
        sex_code = 0 if "Female" in sex else 1
        family_size = sibsp + parch + 1
        is_alone = 1 if family_size == 1 else 0

        input_data = {
            'Pclass': pclass,
            'Sex': sex_code,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'FamilySize': family_size,
            'IsAlone': is_alone,
            'Embarked_Q': 1 if "Queenstown" in embarked else 0,
            'Embarked_S': 1 if "Southampton" in embarked else 0
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure all features are present
        for feature in features:
            if feature not in input_df.columns:
                input_df[feature] = 0

        # Reorder to match training
        input_df = input_df[features]

        # ========== MAKE PREDICTION ==========
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

    # ========== DISPLAY RESULTS ==========
    st.header("🎯 Prediction Results")

    # Result box
    if prediction == 1:
        st.markdown(
            f'<div class="prediction-box survived">'
            f'<h2>✅ YOU SURVIVE THE DISASTER!</h2>'
            f'<p>You would have made it onto a lifeboat</p>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.balloons()
    else:
        st.markdown(
            f'<div class="prediction-box not-survived">'
            f'<h2>❌ YOU PERISH IN THE DISASTER</h2>'
            f'<p>You would not have reached a lifeboat in time</p>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Probability gauge
    st.subheader("Survival Probability")
    col_prob1, col_prob2, col_prob3 = st.columns([2, 1, 1])

    with col_prob1:
        st.progress(float(probability))

    with col_prob2:
        st.metric("Probability", f"{probability:.1%}")

    with col_prob3:
        st.metric("Confidence", f"{(1 - abs(probability - 0.5)) * 2:.0%}")

    # ========== FACTOR ANALYSIS ==========
    st.subheader("📊 Factors Affecting Your Chances")

    factors_col1, factors_col2 = st.columns(2)

    with factors_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)

        # Gender factor
        if "Female" in sex:
            st.markdown("**✅ Gender Advantage**")
            st.write("Women had 74% survival rate vs 19% for men")
            st.success("+30% survival chance")
        else:
            st.markdown("**❌ Gender Disadvantage**")
            st.write("'Women and children first' policy")
            st.error("-30% survival chance")

        st.markdown('</div>', unsafe_allow_html=True)

        # Class factor
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)

        if pclass == 1:
            st.markdown("**✅ First Class Priority**")
            st.write("Closest to lifeboats (63% survival)")
            st.success("+20% survival chance")
        elif pclass == 2:
            st.markdown("**⚠️ Second Class**")
            st.write("Moderate access to lifeboats (43% survival)")
            st.warning("Neutral effect")
        else:
            st.markdown("**❌ Third Class Disadvantage**")
            st.write("Far from lifeboat decks (25% survival)")
            st.error("-20% survival chance")

        st.markdown('</div>', unsafe_allow_html=True)

    with factors_col2:
        # Age factor
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)

        if age < 18:
            st.markdown("**✅ Child Priority**")
            st.write("Children were prioritized (52% survival)")
            st.success("+15% survival chance")
        elif age > 50:
            st.markdown("**⚠️ Elderly**")
            st.write("Lower mobility affected chances")
            st.warning("-10% survival chance")
        else:
            st.markdown("**⚖️ Adult**")
            st.write("Standard survival rate")
            st.info("Neutral effect")

        st.markdown('</div>', unsafe_allow_html=True)

        # Family factor
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)

        if is_alone:
            st.markdown("**⚠️ Traveling Alone**")
            st.write("No family assistance during evacuation")
            st.warning("-5% survival chance")
        else:
            st.markdown("**✅ With Family**")
            st.write("Family could help reach lifeboats")
            st.success("+5% survival chance")

        st.markdown('</div>', unsafe_allow_html=True)

    # ========== HISTORICAL CONTEXT ==========
    st.markdown("---")
    st.header("📜 Historical Context")

    hist_col1, hist_col2, hist_col3, hist_col4 = st.columns(4)

    with hist_col1:
        st.metric("Total Passengers", "2,224")
        st.caption("Aboard Titanic")

    with hist_col2:
        st.metric("Survivors", "710")
        st.caption("31.9% survived")

    with hist_col3:
        st.metric("Lifeboats", "20")
        st.caption("Capacity: 1,178 people")

    with hist_col4:
        st.metric("Water Temperature", "-2°C")
        st.caption("Fatal within 15-30 minutes")

# ========== FOOTER ==========
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p><strong>Built with ❤️ using Streamlit & Random Forest Classifier</strong></p>
    <p><em>This is a machine learning prediction based on historical patterns from the 1912 Titanic disaster.</em></p>
    <p><small>Disclaimer: This is an educational demonstration. Actual historical outcomes were complex and tragic.</small></p>
</div>
""", unsafe_allow_html=True)