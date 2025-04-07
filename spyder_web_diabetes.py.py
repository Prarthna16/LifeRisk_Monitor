import streamlit as st
import joblib  # Changed from pickle to joblib
import numpy as np

# Diabetes Prediction Page (No changes needed)
st.title("Diabetes Prediction")
st.write("Enter the details below to predict diabetes:")

heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=180)
bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0)
sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0)
physical_activity = st.number_input("Physical Activity (hours)", min_value=0.0, max_value=24.0)
systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=200)
diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=120)
oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=60, max_value=100)
body_temperature = st.number_input("Body Temperature (°C)", min_value=34.0, max_value=42.0)
frequent_urination = st.selectbox("Frequent Urination", [0, 1])
frequent_fatigue = st.selectbox("Frequent Fatigue", [0, 1])
unexplained_weight_loss = st.selectbox("Unexplained Weight Loss", [0, 1])

if st.button("Predict Diabetes"):
    diabetes_model = joblib.load(r"C:\Users\shail\Downloads\final year project pkl\lgb_diabetes_model.pkl") #joblib
    scaler = joblib.load(r"C:\Users\shail\Downloads\final year project pkl\scaler.pkl") #joblib
    input_data = [[heart_rate, bmi, sleep, physical_activity, systolic_bp, diastolic_bp, oxygen_saturation,
                   body_temperature, frequent_urination, frequent_fatigue, unexplained_weight_loss]]
    input_data_scaled = scaler.transform(input_data)
    prediction = diabetes_model.predict(input_data_scaled)
    if prediction == 1:
        st.write("Prediction: Diabetes Positive")
    else:
        st.write("Prediction: Diabetes Negative")

# Heart Disease Prediction Page (Changes here!)
st.title("Heart Disease Prediction")
st.write("Enter the details below to predict heart disease:")

age = st.number_input("Age", min_value=18, max_value=120)
gender = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
resting_bp = st.number_input("Resting BP", min_value=80, max_value=200)
resting_hr = st.number_input("Resting HR", min_value=60, max_value=180)
chest_pain_type = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
exercise_angina = st.selectbox("Exercise Angina (0: No, 1: Yes)", [0, 1])
physical_activity_level = st.number_input("Physical Activity Level (0-10)", min_value=0, max_value=10)
smoking_status = st.selectbox("Smoking Status (0: No, 1: Yes)", [0, 1])
family_history = st.selectbox("Family History (0: No, 1: Yes)", [0, 1])
bmi = st.number_input("BMI", min_value=18.0, max_value=50.0)
diabetes = st.selectbox("Diabetes (0: No, 1: Yes)", [0, 1])

if st.button("Predict Heart Disease"):
    heart_model = joblib.load(r"C:\Users\shail\Downloads\final year project pkl\heart_lgb_updated.pkl") #joblib
    scaler = joblib.load(r"C:\Users\shail\Downloads\final year project pkl\scaler_heart_updated.pkl") #joblib

    # Corrected input data order!
    input_data = [[age, gender, resting_bp, resting_hr, chest_pain_type, exercise_angina, physical_activity_level, smoking_status, family_history, bmi, diabetes]]

    input_data_scaled = scaler.transform(input_data)
    prediction_prob = heart_model.predict_proba(input_data_scaled)[0][1]
    threshold = 0.5
    if prediction_prob >= threshold:
        st.write("Prediction: Heart Disease Positive")
    else:
        st.write("Prediction: Heart Disease Negative")
    st.write(f"Prediction Probability: {prediction_prob:.2f}")









