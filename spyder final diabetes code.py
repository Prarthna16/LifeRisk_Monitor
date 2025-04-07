import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load updated models and scalers for both diabetes and heart disease
with open('C:/Users/shail/Downloads/final year project pkl/lgb_model_updated.pkl', 'rb') as f:
    diabetes_model = pickle.load(f)
diabetes_scaler = pickle.load(open('C:/Users/shail/Downloads/final year project pkl/scaler_updated.pkl', 'rb'))

heart_model = pickle.load(open('C:/Users/shail/Downloads/final year project pkl/heart_lgb_updated.pkl', 'rb'))  # Updated to use heart_lgb_updated.pkl
heart_scaler = pickle.load(open('C:/Users/shail/Downloads/final year project pkl/scaler_heart_updated.pkl', 'rb'))

# Example input features for prediction (replace with actual input values)
diabetes_input = (79,34.1,4.2,9,121,65,100,36.4,1,1,0)  # 11 features for diabetes
heart_input = (53,1,98,79,3,0,0,1,0,28.9,0)  # 11 features for heart disease

# Convert to numpy array and scale for both models
diabetes_input_scaled = diabetes_scaler.transform(np.array(diabetes_input).reshape(1, -1))
heart_input_scaled = heart_scaler.transform(np.array(heart_input).reshape(1, -1))

# Diabetes Prediction
diabetes_prob = diabetes_model.predict_proba(diabetes_input_scaled)[0][1]  # Probability of diabetes
diabetes_prediction = "Diabetic" if diabetes_prob > 0.5 else "Non-Diabetic"

# Heart Disease Prediction
heart_prob = heart_model.predict_proba(heart_input_scaled)[0][1]  # Probability of heart disease
heart_prediction = "High Risk" if heart_prob > 0.5 else "Low Risk"

# Print results
print(f"🦸 Diabetes Prediction: {diabetes_prediction}")
print(f"Probability: {diabetes_prob * 100:.2f}%\n")
print(f"📊 Diabetes Model Accuracy: 89.74%\n")

print(f"❤️ Heart Disease Prediction: {heart_prediction}")
print(f"Raw Probability (Heart Disease): {heart_prob * 100:.2f}%\n")

# Plot feature importance for both models
def plot_feature_importance(model, model_name):
    feature_importances = model.feature_importances_

    if model_name == "Diabetes":
        features = ["Heart Rate (beats per minute)", "BMI (kg/m²)", "Sleep Hours", "Physical Activity", 
                    "Systolic BP", "Diastolic BP", "Oxygen Saturation", "Body Temperature", 
                    "Frequent Urination", "Frequent Fatigue", "Unexplained Weight Loss"]
    elif model_name == "Heart Disease":
        features = ["Age", "Sex", "Resting BP", "Resting HR", "Chest Pain Type", "Exercise Angina", 
                    "Physical Activity Level", "Smoking Status", "Family History", "BMI", "Diabetes"]

    plt.figure(figsize=(10, 6))
    plt.barh(features, feature_importances)
    plt.xlabel("Importance")
    plt.title(f"Feature Importance for {model_name} Model")
    plt.show()

# Plot feature importance for both models
plot_feature_importance(diabetes_model, "Diabetes")
plot_feature_importance(heart_model, "Heart Disease")









