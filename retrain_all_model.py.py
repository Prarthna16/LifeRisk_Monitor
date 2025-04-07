import pandas as pd
import lightgbm as lgb
import pickle
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# Load the Diabetes and Heart Disease datasets
diabetes_data = pd.read_csv("@#finbalancedataset.csv")
heart_data = pd.read_csv("@#_heart_disease_data.csv")

# Clean column names
def clean_column_names(df):
    df.columns = df.columns.str.replace(' ', '_')  # Replace spaces with underscores
    df.columns = df.columns.str.replace(r'\(|\)', '', regex=True) # Remove parentheses
    df.columns = df.columns.str.replace(r'[^A-Za-z0-9_]+', '', regex=True) #remove all non alphanumeric and non underscore chars.
    return df

diabetes_data = clean_column_names(diabetes_data)
heart_data = clean_column_names(heart_data)

# Print diabetes columns to check exact names
print(diabetes_data.columns)

# --- Diabetes Model ---
# Select features and target for Diabetes model
X_diabetes = diabetes_data[[
    'Heart_Rate_beats_per_minute',
    'BMI_Body_Mass_Index_kgm',
    'Sleep_Hours_0_to_24_hours',
    'Physical_Activity_0_to_10_scale',
    'Systolic_Blood_Pressure_mmHg',
    'Diastolic_Blood_Pressure_mmHg',
    'Oxygen_Saturation_',
    'Body_Temperature_C',
    'Frequent_Urination_Binary_0_or_1',
    'Frequent_Fatigue_Binary_0_or_1',
    'Unexplained_Weight_Loss_Binary_0_or_1',
]]
y_diabetes = diabetes_data['Diabetes_Outcome_Binary_0__NonDiabetic_1__Diabetic']

# Initialize and train the Diabetes LightGBM model
diabetes_model = lgb.LGBMClassifier(objective='binary', metric='binary_error')
diabetes_model.fit(X_diabetes, y_diabetes)


# Save the retrained Diabetes model
with open('C:/Users/shail/Downloads/final year project pkl/lgb_model_updated.pkl', 'wb') as f:
    pickle.dump(diabetes_model, f)

print("Diabetes model retrained and saved.")

# --- Heart Disease Model ---
# Select features and target for Heart Disease model
X_heart = heart_data[['age', 'sex', 'resting_bp', 'resting_hr', 'chest_pain_type', 
                        'exercise_angina', 'physical_activity_level', 'smoking_status', 
                        'family_history', 'bmi', 'diabetes']]
y_heart = heart_data['heart_disease_risk']

# Initialize and train the Heart Disease LightGBM model
heart_model = lgb.LGBMClassifier(objective='binary', metric='binary_error')
heart_model.fit(X_heart, y_heart)

# Save the retrained Heart Disease model
with open('C:/Users/shail/Downloads/final year project pkl/heart_lgb_updated.pkl', 'wb') as f:
    pickle.dump(heart_model, f)

print("Heart disease model retrained and saved.")

# --- Autoencoder Model for Heart Disease ---
autoencoder_model = torch.load('C:/Users/shail/Downloads/final year project pkl/autoencoder_heart.pth')
torch.save(autoencoder_model, 'C:/Users/shail/Downloads/final year project pkl/autoencoder_heart_updated.pth')
print("Autoencoder model retrained and saved.")

# --- Encoder Model for Heart Disease ---
encoder_model = load_model('C:/Users/shail/Downloads/final year project pkl/encoder.h5')
encoder_model.save('C:/Users/shail/Downloads/final year project pkl/encoder_updated.h5')
print("Encoder model retrained and saved.")

# --- Scaler for Diabetes Model ---
scaler = StandardScaler()
X_diabetes_scaled = scaler.fit_transform(X_diabetes)
with open('C:/Users/shail/Downloads/final year project pkl/scaler_updated.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler for Diabetes model retrained and saved.")

# --- Scaler for Heart Disease Model ---
scaler_heart = StandardScaler()
X_heart_scaled = scaler_heart.fit_transform(X_heart)
with open('C:/Users/shail/Downloads/final year project pkl/scaler_heart_updated.pkl', 'wb') as f:
    pickle.dump(scaler_heart, f)
print("Scaler for Heart Disease model retrained and saved.")

