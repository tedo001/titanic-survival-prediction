@echo off
echo ========================================
echo TITANIC SURVIVAL PREDICTION SETUP
echo ========================================

echo Creating folders...
mkdir data 2>nul
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir models 2>nul
mkdir reports 2>nul

echo Installing packages...
pip install streamlit pandas numpy scikit-learn joblib

echo Training model...
python simple_train.py

echo Starting Streamlit app...
echo.
echo The app will open in your browser at:
echo http://localhost:8501
echo.
echo Press any key to continue...
pause >nul

streamlit run simple_app.py