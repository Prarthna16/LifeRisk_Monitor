import joblib
import os

# Full path to the model
model_path = "C:\\Users\\shail\\Downloads\\final year project pkl\\boosted_rf_model.pkl"

try:
    # Load the model using the full path
    diabetes_model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
