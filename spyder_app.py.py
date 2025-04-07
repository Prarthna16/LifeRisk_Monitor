import streamlit as st
import joblib
import numpy as np
import pandas as pd
import random

# --- Backend Functions (No Changes) ---
def diabetes_prediction(input_data):
    diabetes_model = joblib.load(r"C:\Users\shail\Downloads\final year project pkl\lgb_model_updated.pkl")
    scaler = joblib.load(r"C:\Users\shail\Downloads\final year project pkl\scaler_updated.pkl")
    input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    prediction = diabetes_model.predict(input_data_scaled)[0]
    prob_diabetic = diabetes_model.predict_proba(input_data_scaled)[0][1] if hasattr(diabetes_model, "predict_proba") else None
    return ("Diabetic" if prediction == 1 else "Non-Diabetic"), prob_diabetic

def heart_prediction(input_data):
    heart_model = joblib.load(r"C:\Users\shail\Downloads\final year project pkl\heart_lgb_updated.pkl")
    scaler = joblib.load(r"C:\Users\shail\Downloads\final year project pkl\scaler_heart_updated.pkl")
    input_data_scaled = scaler.transform(np.array(input_data).reshape(1, -1))
    prediction_prob = heart_model.predict_proba(input_data_scaled)[0][1]
    return "High Risk" if prediction_prob >= 0.5 else "Low Risk"

# --- Streamlit Frontend ---
st.set_page_config(page_title="LifeRisk Monitor", page_icon="❤️", layout="wide")

# Enhanced CSS for Modern Medical Theme
st.markdown(
    """
    <style>
    /* Main Theme Colors */
    :root {
        --primary: #008080;       /* Teal */
        --primary-light: #20B2AA; /* Light Sea Green */
        --accent: #00CED1;        /* Dark Turquoise */
        --success: #57CC99;       /* Green for positive results */
        --warning: #FFB703;       /* Warning yellow */
        --danger: #E63946;        /* Alert red */
        --light-bg: #E0F2F1;      /* Pastel Teal */
        --dark-text: #1A1A2E;     /* Dark text */
        --medium-text: #495057;   /* Medium text */
        --light-text: #6C757D;    /* Light text */
        --card-bg: #FFFFFF;       /* Card background */
        --border-color: #DEE2E6;  /* Border color */
        --secondary-bg: #B2DFDB;  /* Light Pastel Teal */
    }
    
    /* Base Styles */
    body {
        font-family: 'Poppins', 'Roboto', sans-serif;
        background-color: var(--light-bg);
        color: var(--dark-text);
        line-height: 1.7;
    }
    
    .stApp {
        background: #E0F2F1;  /* Plain soft teal background */
    }
    
    /* Remove overlay */
    .stApp::before {
        display: none;
    }
    
    /* Add subtle medical pattern */
    .stApp::after {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%230077B6' fill-opacity='0.05'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
        pointer-events: none;
        z-index: -2;
    }
    
    /* Typography */
    h1, h2, h3, .big-font {
        font-weight: 700;
        color: var(--primary);
        margin-bottom: 0.5rem;
    }
    
    .big-font {
        font-size: 2.5rem !important;
        letter-spacing: -0.5px;
        line-height: 1.2;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary-light);
        margin-bottom: 1.5rem;
    }
    
    .medium-font {
        font-size: 1rem !important;
        line-height: 1.7;
        color: var(--medium-text);
    }
    
    .card-title {
        font-size: 1.25rem !important;
        font-weight: 600;
        color: var(--primary);
        margin-bottom: 0.75rem;
    }
    
    /* Cards and Containers */
    .content-card {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 1px 3px rgba(0, 0, 0, 0.1);
        border-top: 4px solid var(--primary);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        backdrop-filter: blur(5px);
    }
    
    .content-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    
    .result-box {
        background: linear-gradient(145deg, var(--card-bg), var(--secondary-bg));
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 5px solid var(--primary);
    }
    
    .result-box .big-font {
        margin-bottom: 0;
        padding-bottom: 0;
        border-bottom: none;
    }
    
    .image-container {
        display: flex;
        align-items: flex-start;
        margin-bottom: 1.5rem;
        background-color: var(--card-bg);
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .image-container img {
        width: 300px;
        height: auto;
        object-fit: cover;
        border-radius: 8px;
        margin-right: 1.5rem;
    }
    
    .recipe-container {
        display: flex;
        flex-direction: row;
        margin-bottom: 1.5rem;
        background-color: var(--card-bg);
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: transform 0.3s ease;
    }
    
    .recipe-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Form Elements */
    .stButton button {
        background-color: var(--primary) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stButton button:hover {
        background-color: var(--accent) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15) !important;
    }
    
    .stSelectbox, .stNumberInput {
        margin-bottom: 1rem;
    }
    
    .stSelectbox > div > div > div {
        background-color: var(--card-bg);
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div > div:hover {
        border-color: var(--primary) !important;
    }
    
    .stNumberInput input {
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        transition: all 0.3s ease;
        padding: 0.5rem 1rem !important;
    }
    
    .stNumberInput input:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 2px rgba(0, 128, 128, 0.2) !important;
    }
    
    /* Sidebar Styling */
    .css-1d391kg, .css-1lcbmhc {
        background-image: linear-gradient(to bottom, var(--primary), #023E8A);
    }
    
    .sidebar .sidebar-content {
        background-color: transparent;
    }
    
    [data-testid="stSidebar"] {
        background-image: linear-gradient(to bottom, var(--primary), #023E8A);
        color: white;
    }
    
    [data-testid="stSidebar"] .stRadio label, [data-testid="stSidebar"] .stSelectbox label {
        color: white !important;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="stSidebar"] .stRadio div {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    [data-testid="stSidebar"] .css-16huue1 {
        font-size: 1.2rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        padding: 1rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1rem;
    }
    
    /* Medical Icons and Indicators */
    .health-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
        color: var(--primary);
    }
    
    .health-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
    
    .indicator-good {
        background-color: var(--success);
    }
    
    .indicator-warning {
        background-color: var(--warning);
    }
    
    .indicator-danger {
        background-color: var(--danger);
    }
    
    /* Feature Sections */
    .feature-section {
        border-left: 4px solid var(--primary);
        padding-left: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Animation for Loading */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .loading-pulse {
        animation: pulse 1.5s infinite;
    }
    
    /* Medical Dashboard Look */
    .dashboard-stats {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        flex: 1;
        background-color: var(--card-bg);
        border-radius: 10px;
        padding: 1.5rem;
        margin-right: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        text-align: center;
        border-bottom: 3px solid var(--primary);
    }
    
    .stat-card:last-child {
        margin-right: 0;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary);
    }
    
    .stat-label {
        color: var(--light-text);
        font-size: 0.9rem;
    }
    
    /* Tables for Data */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 1.5rem;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    .styled-table thead {
        background-color: var(--primary);
        color: white;
    }
    
    .styled-table th, .styled-table td {
        padding: 1rem;
        text-align: left;
        border-bottom: 1px solid var(--border-color);
    }
    
    .styled-table tbody tr:nth-child(even) {
        background-color: var(--secondary-bg);
    }
    
    /* Page Header */
    .page-header {
        background: linear-gradient(135deg, rgba(0, 128, 128, 0.1) 0%, rgba(32, 178, 170, 0.1) 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .logo-container {
        margin-right: 1rem;
    }
    
    .logo {
        font-size: 2rem;
        color: var(--primary);
    }
    
    /* Utilities */
    .text-center {
        text-align: center;
    }
    
    .mt-4 {
        margin-top: 1.5rem;
    }
    
    .mb-4 {
        margin-bottom: 1.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Header Navigation with improved styling
with st.sidebar:
    st.markdown("<h2 style='color: white; text-align: center; margin-bottom: 2rem;'>LifeRisk Monitor</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: center; margin-bottom: 2rem;'>❤️ 🩺 🫀</div>", unsafe_allow_html=True)
    menu = ["Home", "About Diabetes", "Take Test", "Ayurvedic Solutions", "Heart Health", "Healthy Recipes"]
    choice = st.selectbox("Navigation", menu)
    
    if choice == "Take Test":
        test_choice = st.radio("Select Test Type", ["Diabetes Risk", "Heart Risk"])

# Home Page
if choice == "Home":
    st.markdown("<div class='page-header'><div class='logo-container'>❤️</div><h1 class='big-font'>Welcome to LifeRisk Monitor</h1></div>", unsafe_allow_html=True)
    st.image(r"C:\Users\shail\Downloads\final year project pkl\health_banner.jpg", use_container_width=True)
    
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>Your Digital Health Companion</h2>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>Welcome to LifeRisk Monitorr, your comprehensive digital companion for navigating the complexities of diabetes and heart health. We understand that managing these interconnected conditions requires a holistic approach, which is why our platform seamlessly blends the precision of modern risk assessment tools with the time-honored wisdom of Ayurveda.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Dashboard Stats
    st.markdown("<div class='dashboard-stats'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='stat-card' style='background: linear-gradient(145deg, #E6F7FF, #B3E0FF); border-bottom: 3px solid #0077B6;'>
            <div class='stat-value' style='color: #0077B6;'>Ayurvedic Solutions</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='stat-card' style='background: linear-gradient(145deg, #F0FFF0, #C1FFC1); border-bottom: 3px solid #57CC99;'>
            <div class='stat-value' style='color: #57CC99;'>Take Diabetes and Heart test</div>
        
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='stat-card' style='background: linear-gradient(145deg, #FFF5E6, #FFE0B3); border-bottom: 3px solid #FFB703;'>
            <div class='stat-value' style='color: #FFB703;'>Healthy   Recipes</div>
       
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Features Section
    st.markdown("<h2 class='card-title mt-4'>Key Features</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
        st.markdown("<h3>🔍 Advanced Risk Assessment</h3>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>Our platform uses machine learning algorithms to provide accurate predictions of diabetes and heart disease risk based on your health parameters.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
        st.markdown("<h3>🌿 Ayurvedic Wisdom</h3>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>Access traditional Ayurvedic remedies and practices that have been used for centuries to manage diabetes and heart conditions.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
        st.markdown("<h3>🍽️ Nutritional Guidance</h3>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>Discover heart-healthy and diabetes-friendly recipes that are both delicious and beneficial for your health.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
        st.markdown("<h3>🧘 Holistic Approach</h3>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>Learn exercises and yoga practices specifically designed to improve heart health and manage diabetes.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# About Diabetes
elif choice == "About Diabetes":
    st.markdown("<div class='page-header'><div class='logo-container'>🩺</div><h1 class='big-font'>Understanding Diabetes</h1></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='card-title'>What is Diabetes?</h2>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>Diabetes is a multifaceted, chronic metabolic disorder characterized by the persistent elevation of blood glucose levels, a condition known as hyperglycemia. This occurs due to either the body's inability to produce sufficient insulin, a hormone secreted by the pancreas essential for regulating blood sugar, or when the body's cells become resistant to the insulin that is produced.</p>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>Insulin acts as a key, unlocking cells to allow glucose from the bloodstream to enter and be used for energy. Without adequate insulin or with insulin resistance, glucose accumulates in the blood, leading to a cascade of potential health issues.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.image(r"C:\Users\shail\Downloads\final year project pkl\diabetes_info.jpg", use_container_width=True)

    # Types of Diabetes 
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>Types of Diabetes</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3>Type 1 Diabetes</h3>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>• Autoimmune condition<br>• Body doesn't produce insulin<br>• Usually diagnosed in children and young adults<br>• Requires insulin therapy</p>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3>Type 2 Diabetes</h3>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>• Body becomes resistant to insulin<br>• Most common form of diabetes<br>• Often associated with lifestyle factors<br>• Can be managed with diet, exercise, and medication</p>", unsafe_allow_html=True)

    st.markdown("<h3>Gestational Diabetes</h3>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>Develops during pregnancy and typically resolves after delivery, though it increases the risk of developing type 2 diabetes later in life.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Symptoms and Complications
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>Symptoms & Complications</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3>Common Symptoms</h3>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>• Frequent urination<br>• Excessive thirst<br>• Unexplained weight loss<br>• Fatigue<br>• Blurred vision<br>• Slow-healing wounds</p>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3>Potential Complications</h3>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>• Cardiovascular disease<br>• Nerve damage (neuropathy)<br>• Kidney damage (nephropathy)<br>• Eye damage (retinopathy)<br>• Foot damage<br>• Hearing impairment</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Management
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>Management Strategies</h2>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>Managing diabetes involves a comprehensive approach including:</p>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>• Regular blood glucose monitoring<br>• Balanced diet with controlled carbohydrate intake<br>• Regular physical activity<br>• Medication or insulin therapy as prescribed<br>• Regular medical check-ups<br>• Stress management<br>• Adequate sleep<br>• Avoiding tobacco and limiting alcohol</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Take Test (Diabetes and Heart)
elif choice == "Take Test":
    if test_choice == "Diabetes Risk":
        st.markdown("<div class='page-header'><div class='logo-container'>🔍</div><h1 class='big-font'>Diabetes Risk Assessment</h1></div>", unsafe_allow_html=True)

        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>Please enter your health parameters below for a diabetes risk assessment. This tool uses machine learning to analyze your data and predict your risk of developing diabetes.</p>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=180, value=75, help="Normal resting heart rate for adults ranges from 60 to 100 beats per minute")
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=24.5, help="BMI is a measure of body fat based on height and weight")
            sleep = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, help="Average number of hours you sleep per day")
            physical_activity = st.number_input("Physical Activity (hours/week)", min_value=0.0, max_value=24.0, value=3.0, help="Average hours of moderate to vigorous physical activity per week")
            systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=80, max_value=200, value=120, help="The top number in a blood pressure reading")

        with col2:
            diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=120, value=80, help="The bottom number in a blood pressure reading")
            oxygen_saturation = st.number_input("Oxygen Saturation (%)", min_value=60, max_value=100, value=98, help="The percentage of oxygen in your blood")
            body_temperature = st.number_input("Body Temperature (°C)", min_value=34.0, max_value=42.0, value=36.6, help="Normal body temperature is around 37°C")
            frequent_urination = st.selectbox("Frequent Urination", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Do you urinate more frequently than normal?")
            frequent_fatigue = st.selectbox("Frequent Fatigue", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Do you often feel unusually tired?")

        unexplained_weight_loss = st.selectbox("Unexplained Weight Loss", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Have you lost weight without trying?")

        predict_btn = st.button("Predict Diabetes Risk")

        if predict_btn:
            st.markdown("<div class='loading-pulse' style='text-align: center; margin: 20px 0;'>Analyzing your health data...</div>", unsafe_allow_html=True)
            input_data = [heart_rate, bmi, sleep, physical_activity, systolic_bp, diastolic_bp, oxygen_saturation, body_temperature, frequent_urination, frequent_fatigue, unexplained_weight_loss]

            try:
                if not all(isinstance(x, (int, float)) for x in input_data[:8]):
                    st.error("Please enter valid numerical values.")
                else:
                    result, prob = diabetes_prediction(input_data)
                    
                    if result == "Diabetic":
                        result_color = "#E63946"  # red for diabetic
                    else:
                        result_color = "#57CC99"  # green for non-diabetic
                    
                    st.markdown(f"""
                    <div class='result-box'>
                        <h2 style='color: {result_color};'>Prediction: {result}</h2>
                        <p class='medium-font'>This assessment is based on the information you provided and should be considered as a screening tool only. Please consult with a healthcare professional for a thorough evaluation.</p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Health Tips
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='card-title'>Diabetes Prevention Tips</h2>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>• Maintain a healthy weight<br>• Exercise regularly<br>• Eat a balanced diet rich in fruits, vegetables, and whole grains<br>• Limit processed foods and added sugars<br>• Stay hydrated<br>• Get regular check-ups</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif test_choice == "Heart Risk":
        st.markdown("<div class='page-header'><div class='logo-container'>❤️</div><h1 class='big-font'>Heart Risk Assessment</h1></div>", unsafe_allow_html=True)

        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>Please enter your health parameters below for a heart risk assessment. This tool uses machine learning to analyze your data and predict your risk of developing heart disease.</p>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=120, value=45, help="Your current age in years")
            gender = st.selectbox("Sex", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
            resting_bp = st.number_input("Resting BP (mmHg)", min_value=80, max_value=200, value=120, help="Your blood pressure when you're at rest")
            resting_hr = st.number_input("Resting HR (bpm)", min_value=60, max_value=180, value=75, help="Your heart rate when you're at rest")
            chest_pain_type = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: {0: "Typical Angina", 1: "Atypical Angina", 2: "Non-anginal Pain", 3: "Asymptomatic"}[x], help="Type of chest pain experienced")

        with col2:
            exercise_angina = st.selectbox("Exercise Angina (0: No, 1: Yes)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Do you experience angina during exercise?")
            physical_activity_level = st.number_input("Physical Activity Level (0-10)", min_value=0, max_value=10, value=5, help="Rate your physical activity level")
            smoking_status = st.selectbox("Smoking Status (0: No, 1: Yes)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Do you smoke?")
            family_history = st.selectbox("Family History of Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Is there a family history of heart disease?")
            bmi = st.number_input("BMI", min_value=18.0, max_value=50.0, value=24.5, help="Body Mass Index")

        diabetes = st.selectbox("Diabetes (0: No, 1: Yes)", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", help="Do you have diabetes?")

        if st.button("Predict Heart Risk"):
            input_data = [age, gender, resting_bp, resting_hr, chest_pain_type, exercise_angina, physical_activity_level, smoking_status, family_history, bmi, diabetes]
            st.markdown("<div class='loading-pulse' style='text-align: center; margin: 20px 0;'>Analyzing your cardiac data...</div>", unsafe_allow_html=True)

            try:
                if not all(isinstance(x, (int, float)) for x in input_data[:5]):
                    st.error("Please enter valid numerical values.")
                else:
                    result = heart_prediction(input_data)

                    if result == "High Risk":
                        result_color = "#E63946"  # red for high risk
                    else:
                        result_color = "#57CC99"  # green for low risk

                    st.markdown(f"""
                    <div class='result-box'>
                        <h2 style='color: {result_color};'>Prediction: {result}</h2>
                        <p class='medium-font'>This assessment is based on the information you provided and should be considered as a screening tool only. Please consult with a healthcare professional for a thorough evaluation.</p>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

        # Health Tips for Heart
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='card-title'>Heart Health Tips</h2>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>• Maintain a healthy blood pressure<br>• Keep cholesterol levels in check<br>• Exercise regularly<br>• Eat a heart-healthy diet<br>• Limit sodium intake<br>• Avoid smoking<br>• Manage stress<br>• Get adequate sleep</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Ayurvedic Solutions
elif choice == "Ayurvedic Solutions":
    st.markdown("<div class='page-header'><div class='logo-container'>🌿</div><h1 class='big-font'>Ayurvedic Approach to Health</h1></div>", unsafe_allow_html=True)

    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>Introduction to Ayurveda</h2>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>Ayurveda, meaning 'knowledge of life,' is an ancient Indian system of medicine dating back over 5,000 years. It emphasizes balance between body, mind, and spirit, and focuses on prevention rather than cure. According to Ayurveda, each person has a unique constitution or 'dosha' profile that influences their physical and mental characteristics.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Doshas Section
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>Understanding Doshas</h2>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<h3>Vata</h3>", unsafe_allow_html=True)
        st.image(r"C:\Users\shail\Downloads\final year project pkl\vata_symbol.jpeg", width=100)
        st.markdown("<p class='medium-font'>Represents air and space elements. People with dominant Vata tend to be thin, creative, and energetic when balanced, but anxious and irregular when imbalanced.</p>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3>Pitta</h3>", unsafe_allow_html=True)
        st.image(r"C:\Users\shail\Downloads\final year project pkl\pitta_symbol.jpeg", width=100)
        st.markdown("<p class='medium-font'>Represents fire and water elements. Pitta-dominant individuals are often athletic, intelligent, and driven when balanced, but irritable and inflamed when imbalanced.</p>", unsafe_allow_html=True)

    with col3:
        st.markdown("<h3>Kapha</h3>", unsafe_allow_html=True)
        st.image(r"C:\Users\shail\Downloads\final year project pkl\kapha_symbol.jpeg", width=100)
        st.markdown("<p class='medium-font'>Represents earth and water elements. Kapha types are typically strong, calm, and loyal when balanced, but can become overweight and lethargic when imbalanced.</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Ayurvedic Solutions for Diabetes
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>Ayurvedic Approaches for Diabetes (Madhumeha)</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])

    with col1:
       st.markdown("<h3>Herbal Remedies</h3>", unsafe_allow_html=True)

    # Bitter Gourd
    st.image(r"C:\Users\shail\Downloads\final year project pkl\bittergourd.jpg", width=100)
    st.markdown("<p class='medium-font'><strong>Bitter Gourd (Karela):</strong> Contains insulin-like compounds that help reduce blood glucose levels.</p>", unsafe_allow_html=True)

    # Fenugreek
    st.image(r"C:\Users\shail\Downloads\final year project pkl\fenugreek.jpg", width=100)
    st.markdown("<p class='medium-font'><strong>Fenugreek (Methi):</strong> Seeds contain fiber that helps slow digestion and absorption of carbohydrates.</p>", unsafe_allow_html=True)

    # Turmeric
    st.image(r"C:\Users\shail\Downloads\final year project pkl\turmeric.jpg", width=100)
    st.markdown("<p class='medium-font'><strong>Turmeric:</strong> Contains curcumin, which may help improve insulin sensitivity.</p>", unsafe_allow_html=True)

    # Gymnema Sylvestre
    st.image(r"C:\Users\shail\Downloads\final year project pkl\gymnema.jpeg", width=100)
    st.markdown("<p class='medium-font'><strong>Gymnema Sylvestre (Gurmar):</strong> Known as 'sugar destroyer,' it may help reduce sugar cravings and improve glucose uptake.</p>", unsafe_allow_html=True)


   
    st.markdown("<h3>Dietary Recommendations</h3>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>• Favor bitter, astringent, and pungent tastes<br>• Include barley, oats, and millet<br>• Consume plenty of green vegetables<br>• Include amla (Indian gooseberry), which is high in vitamin C<br>• Limit sweet, sour, and salty tastes<br>• Avoid processed foods, refined sugars, and carbohydrates</p>", unsafe_allow_html=True)

    st.markdown("<h3>Lifestyle Practices</h3>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>• Regular physical activity like yoga and walking<br>• Practice pranayama (breathing exercises) daily<br>• Maintain regular eating and sleeping schedule<br>• Stress management through meditation</p>", unsafe_allow_html=True)

    with col2:
        st.image(r"C:\Users\shail\Downloads\final year project pkl\ayurveda.jpg", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Ayurvedic Solutions for Heart Health
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>Ayurvedic Approaches for Heart Health (Hridroga)</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(r"C:\Users\shail\Downloads\final year project pkl\heart_ayurveda.jpg", use_container_width=True)

    with col2:
        st.markdown("<h3>Herbal Remedies</h3>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'><strong>Arjuna:</strong> Bark extract has cardioprotective properties.<br><strong>Garlic:</strong> Helps reduce cholesterol and blood pressure.<br><strong>Holy Basil (Tulsi):</strong> Reduces stress and supports cardiovascular function.<br><strong>Guggulu:</strong> Helps lower cholesterol and supports healthy weight.</p>", unsafe_allow_html=True)

        st.markdown("<h3>Dietary Recommendations</h3>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>• Include heart-healthy fats like ghee in moderation<br>• Consume plenty of fresh fruits and vegetables<br>• Include whole grains and legumes<br>• Add spices like turmeric, ginger, and cinnamon<br>• Limit salt, processed foods, and heavy meals</p>", unsafe_allow_html=True)

        st.markdown("<h3>Yoga for Heart Health</h3>", unsafe_allow_html=True)
        st.image(r"C:\Users\shail\Downloads\final year project pkl\yoga.jpg", use_container_width=True)
        st.markdown("<p class='medium-font'>• Vajrasana (Thunderbolt Pose)<br>• Dhanurasana (Bow Pose)<br>• Pawanmuktasana (Wind-Relieving Pose)<br>• Shavasana (Corpse Pose)<br>• Heart-opening poses like Bhujangasana (Cobra Pose)</p>", unsafe_allow_html=True)

        # Add yoga pose images
        st.markdown("<h4>Yoga Poses Demonstration</h4>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(r"C:\Users\shail\Downloads\final year project pkl\vajrasana.jpg", caption="Vajrasana (Thunderbolt Pose)", use_container_width=True)
            st.image(r"C:\Users\shail\Downloads\final year project pkl\dhanurasana.jpeg", caption="Dhanurasana (Bow Pose)", use_container_width=True)
            st.image(r"C:\Users\shail\Downloads\final year project pkl\pawanmuktasana.jpg", caption="Pawanmuktasana (Wind-Relieving Pose)", use_container_width=True)
        
        with col2:
            st.image(r"C:\Users\shail\Downloads\final year project pkl\shavasana.jpeg", caption="Shavasana (Corpse Pose)", use_container_width=True)
            st.image(r"C:\Users\shail\Downloads\final year project pkl\bhujangasana.jpg", caption="Bhujangasana (Cobra Pose)", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Important Note
    st.markdown("<div class='content-card' style='border-left: 4px solid #FFB703;'>", unsafe_allow_html=True)
    st.markdown("<h3>Important Note</h3>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>While Ayurvedic approaches have been used traditionally for managing diabetes and heart conditions, they should be considered complementary to modern medical treatment. Always consult with your healthcare provider before starting any new treatment or supplement, especially if you have existing health conditions or are taking medications.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Heart Health
elif choice == "Heart Health":
    st.markdown("<div class='page-header'><div class='logo-container'>❤️</div><h1 class='big-font'>Understanding Heart Health</h1></div>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("<div class='content-card'>", unsafe_allow_html=True)
        st.markdown("<h2 class='card-title'>What is Heart Disease?</h2>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>Heart disease, often referred to as cardiovascular disease, encompasses a range of conditions affecting the heart and blood vessels. These conditions include coronary artery disease, heart rhythm problems (arrhythmias), heart valve disease, heart muscle disease (cardiomyopathy), congenital heart defects, and heart infection.</p>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>The most common form is coronary artery disease, which develops when the main blood vessels that supply the heart become damaged or diseased, often due to plaque buildup (atherosclerosis).</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.image(r"C:\Users\shail\Downloads\final year project pkl\heart_anatomy.jpg", use_container_width=True)

    # Risk Factors
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>Risk Factors</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3>Modifiable Risk Factors</h3>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>• High blood pressure<br>• High cholesterol levels<br>• Smoking<br>• Diabetes<br>• Obesity<br>• Physical inactivity<br>• Unhealthy diet<br>• Excessive alcohol consumption<br>• Stress</p>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3>Non-modifiable Risk Factors</h3>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>• Age (risk increases with age)<br>• Gender (men generally at higher risk)<br>• Family history<br>• Ethnicity/Race<br>• Previous history of heart disease</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Warning Signs
    st.markdown("<div class='content-card' style='border-left: 4px solid #E63946;'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>Warning Signs of Heart Attack</h2>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>Recognizing the warning signs of a heart attack can save lives. If you or someone you know experiences these symptoms, seek emergency medical attention immediately:</p>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>• Chest pain or discomfort that may feel like pressure, squeezing, fullness, or pain<br>• Pain or discomfort in other areas of the upper body, including arms, back, neck, jaw, or stomach<br>• Shortness of breath<br>• Cold sweat<br>• Nausea or vomiting<br>• Lightheadedness<br>• Unusual fatigue</p>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>Note: Women may experience different symptoms than men, often including back or jaw pain, shortness of breath, and nausea without significant chest discomfort.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Prevention Strategies
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>Prevention Strategies</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<h3>Lifestyle Modifications</h3>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>• Adopt a heart-healthy diet rich in fruits, vegetables, whole grains, lean proteins, and healthy fats<br>• Exercise regularly (aim for at least 150 minutes of moderate activity per week)<br>• Maintain a healthy weight<br>• Quit smoking and avoid secondhand smoke<br>• Limit alcohol consumption<br>• Manage stress through techniques like meditation, yoga, or deep breathing<br>• Get adequate sleep (7-9 hours for adults)</p>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3>Medical Management</h3>", unsafe_allow_html=True)
        st.markdown("<p class='medium-font'>• Regular health check-ups<br>• Monitor and control blood pressure<br>• Manage cholesterol levels<br>• Control diabetes if present<br>• Take medications as prescribed<br>• Consider preventive aspirin therapy if recommended by your doctor</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Connection Between Heart Disease and Diabetes
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>The Heart-Diabetes Connection</h2>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>Diabetes and heart disease share many risk factors and often occur together. People with diabetes are two to four times more likely to develop heart disease compared to those without diabetes.</p>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>Key connections include:</p>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>• High blood glucose can damage blood vessels and the nerves that control the heart<br>• Diabetes often comes with risk factors like high blood pressure and high cholesterol<br>• Insulin resistance affects the lining of blood vessels, making them more susceptible to atherosclerosis<br>• Both conditions create a state of chronic low-grade inflammation in the body</p>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>Managing diabetes well can significantly reduce your risk of developing heart disease. Similarly, heart-healthy habits can help prevent or manage diabetes.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Healthy Recipes
elif choice == "Healthy Recipes":
    st.markdown("<div class='page-header'><div class='logo-container'>🍽️</div><h1 class='big-font'>Heart & Diabetes Friendly Recipes</h1></div>", unsafe_allow_html=True)

    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<p class='medium-font'>Discover delicious recipes that are specifically designed to support heart health and help manage diabetes. These recipes focus on nutrient-dense ingredients, balanced macronutrients, and minimal added sugars.</p>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Recipe Categories
    recipe_category = st.selectbox("Recipe Category", ["Breakfast", "Lunch", "Dinner", "Snacks", "Desserts"])

    if recipe_category == "Breakfast":
        st.markdown("<h2 class='card-title mt-4'>Healthy Breakfast Recipes</h2>", unsafe_allow_html=True)

        # Recipe 1
        st.markdown("<div class='recipe-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(r"C:\Users\shail\Downloads\final year project pkl\oatmeal.jpg", use_container_width=True)
        with col2:
            st.markdown("<h3>Cinnamon & Berries Overnight Oats</h3>", unsafe_allow_html=True)
            st.markdown("<p><strong>Ingredients:</strong> 1/2 cup rolled oats, 1 tablespoon chia seeds, 3/4 cup unsweetened almond milk, 1/2 teaspoon cinnamon, 1/2 cup mixed berries, 1 tablespoon chopped nuts, 1 teaspoon honey (optional)</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Instructions:</strong> Mix oats, chia seeds, almond milk, and cinnamon in a jar. Refrigerate overnight. In the morning, top with berries and nuts.</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Benefits:</strong> High in soluble fiber which helps manage blood sugar and cholesterol. The berries add antioxidants and natural sweetness without spiking blood sugar.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Recipe 2
        st.markdown("<div class='recipe-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(r"C:\Users\shail\Downloads\final year project pkl\veggie_omelette.jpg", use_container_width=True)
        with col2:
            st.markdown("<h3>Vegetable-Packed Omelette</h3>", unsafe_allow_html=True)
            st.markdown("<p><strong>Ingredients:</strong> 2 eggs, 1/4 cup chopped bell peppers, 1/4 cup spinach, 2 tablespoons chopped onions, 1 tablespoon olive oil, 1/4 avocado (sliced), Salt and pepper to taste</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Instructions:</strong> Whisk eggs with salt and pepper. Sauté vegetables in olive oil until soft. Pour eggs over vegetables and cook until set. Fold and serve with avocado slices.</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Benefits:</strong> Protein-rich breakfast that helps maintain steady blood sugar levels. The vegetables add fiber and nutrients, while healthy fats from olive oil and avocado support heart health.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif recipe_category == "Lunch":
        st.markdown("<h2 class='card-title mt-4'>Healthy Lunch Recipes</h2>", unsafe_allow_html=True)

        # Recipe 1
        st.markdown("<div class='recipe-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(r"C:\Users\shail\Downloads\final year project pkl\quinoa_salad.jpg", use_container_width=True)
        with col2:
            st.markdown("<h3>Mediterranean Quinoa Bowl</h3>", unsafe_allow_html=True)
            st.markdown("<p><strong>Ingredients:</strong> 1/2 cup cooked quinoa, 1 cup mixed greens, 1/4 cup cucumber (diced), 1/4 cup cherry tomatoes (halved), 2 tablespoons red onion (diced), 2 tablespoons kalamata olives, 2 tablespoons feta cheese, 1 tablespoon olive oil, 1 tablespoon lemon juice, 1 teaspoon dried oregano</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Instructions:</strong> Mix olive oil, lemon juice, and oregano to make dressing. Combine all other ingredients in a bowl and drizzle with dressing.</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Benefits:</strong> Quinoa provides complex carbs and protein for sustainable energy. The Mediterranean ingredients are rich in heart-healthy fats and antioxidants.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Recipe 2
        st.markdown("<div class='recipe-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(r"C:\Users\shail\Downloads\final year project pkl\lentil_soup.jpg", use_container_width=True)
        with col2:
            st.markdown("<h3>Spiced Lentil & Vegetable Soup</h3>", unsafe_allow_html=True)
            st.markdown("<p><strong>Ingredients:</strong> 1 cup red lentils, 1 onion (chopped), 2 carrots (diced), 2 celery stalks (diced), 2 cloves garlic (minced), 1 teaspoon cumin, 1/2 teaspoon turmeric, 4 cups vegetable broth, 1 tablespoon olive oil, 1 cup spinach, Salt and pepper to taste</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Instructions:</strong> Sauté onion, carrots, celery, and garlic in olive oil. Add spices and cook for 1 minute. Add lentils and broth, bring to a boil, then simmer for 20 minutes. Add spinach at the end and season to taste.</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Benefits:</strong> Lentils have a low glycemic index and are high in fiber and plant protein. Turmeric and cumin have anti-inflammatory properties beneficial for both diabetes and heart health.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif recipe_category == "Dinner":
        st.markdown("<h2 class='card-title mt-4'>Healthy Dinner Recipes</h2>", unsafe_allow_html=True)

        # Recipe 1
        st.markdown("<div class='recipe-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(r"C:\Users\shail\Downloads\final year project pkl\baked_salmon.jpg", use_container_width=True)
        with col2:
            st.markdown("<h3>Herb-Crusted Baked Salmon with Roasted Vegetables</h3>", unsafe_allow_html=True)
            st.markdown("<p><strong>Ingredients:</strong> 4 oz salmon fillet, 1 tablespoon olive oil, 1 tablespoon fresh herbs (dill, parsley, thyme), 1 clove garlic (minced), 1 cup mixed vegetables (broccoli, bell peppers, zucchini), 1/2 lemon, Salt and pepper to taste</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Instructions:</strong> Preheat oven to 400°F. Mix herbs, garlic, and half the olive oil. Rub on salmon. Toss vegetables with remaining oil. Place salmon and vegetables on baking sheet and roast for 15-18 minutes. Squeeze lemon over before serving.</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Benefits:</strong> Salmon is rich in omega-3 fatty acids that support heart health. Combined with fiber-rich vegetables, this meal has a low glycemic impact while providing essential nutrients.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Recipe 2
        st.markdown("<div class='recipe-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(r"C:\Users\shail\Downloads\final year project pkl\chickpea_curry.jpg", use_container_width=True)
        with col2:
            st.markdown("<h3>Vegetable and Chickpea Curry with Brown Rice</h3>", unsafe_allow_html=True)
            st.markdown("<p><strong>Ingredients:</strong> 1 cup chickpeas (cooked), 1 onion (chopped), 2 cloves garlic (minced), 1 inch ginger (grated), 1 tablespoon curry powder, 1/2 teaspoon turmeric, 1 cup mixed vegetables (cauliflower, peas, carrots), 1 cup low-fat coconut milk, 1/2 cup vegetable broth, 1/2 cup brown rice (cooked), Fresh cilantro</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Instructions:</strong> Sauté onion, garlic, and ginger. Add spices and cook for 1 minute. Add vegetables and cook for 5 minutes. Add chickpeas, coconut milk, and broth. Simmer for 15 minutes. Serve over brown rice and garnish with cilantro.</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Benefits:</strong> Chickpeas have a low glycemic index and are high in protein and fiber. The spices, especially turmeric, have anti-inflammatory properties. Brown rice provides complex carbohydrates that release energy slowly.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif recipe_category == "Snacks":
        st.markdown("<h2 class='card-title mt-4'>Healthy Snack Ideas</h2>", unsafe_allow_html=True)

        # Recipe 1
        st.markdown("<div class='recipe-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(r"C:\Users\shail\Downloads\final year project pkl\greek_yogurt.jpg", use_container_width=True)
        with col2:
            st.markdown("<h3>Greek Yogurt with Nuts and Berries</h3>", unsafe_allow_html=True)
            st.markdown("<p><strong>Ingredients:</strong> 1/2 cup plain Greek yogurt, 1 tablespoon mixed nuts (almonds, walnuts), 1/4 cup mixed berries, 1/2 teaspoon cinnamon, 1 teaspoon chia seeds</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Instructions:</strong> Top Greek yogurt with nuts, berries, cinnamon, and chia seeds. Mix and enjoy.</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Benefits:</strong> Greek yogurt provides protein without spiking blood sugar. Nuts add healthy fats and fiber, while berries offer antioxidants with a low glycemic impact.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Recipe 2
        st.markdown("<div class='recipe-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(r"C:\Users\shail\Downloads\final year project pkl\hummus.jpg", use_container_width=True)
        with col2:
            st.markdown("<h3>Vegetable Sticks with Homemade Hummus</h3>", unsafe_allow_html=True)
            st.markdown("<p><strong>Ingredients:</strong> 1 cup chickpeas (cooked), 2 tablespoons tahini, 1 clove garlic, 2 tablespoons lemon juice, 1 tablespoon olive oil, 1/2 teaspoon cumin, Salt to taste, Assorted vegetable sticks (carrots, cucumbers, bell peppers, celery)</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Instructions:</strong> Blend chickpeas, tahini, garlic, lemon juice, olive oil, cumin, and salt until smooth. Serve with vegetable sticks.</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Benefits:</strong> Chickpeas are high in fiber and protein, which help stabilize blood sugar. The combination with raw vegetables creates a nutrient-dense, heart-healthy snack.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    elif recipe_category == "Desserts":
        st.markdown("<h2 class='card-title mt-4'>Healthy Dessert Options</h2>", unsafe_allow_html=True)

        # Recipe 1
        st.markdown("<div class='recipe-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(r"C:\Users\shail\Downloads\final year project pkl\chia_pudding.jpg", use_container_width=True)
        with col2:
            st.markdown("<h3>Vanilla Chia Seed Pudding</h3>", unsafe_allow_html=True)
            st.markdown("<p><strong>Ingredients:</strong> 2 tablespoons chia seeds, 1/2 cup unsweetened almond milk, 1/4 teaspoon vanilla extract, 1 teaspoon honey (optional), 1/4 cup fresh fruit for topping</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Instructions:</strong> Mix chia seeds, almond milk, vanilla, and honey. Refrigerate for at least 2 hours or overnight. Top with fresh fruit before serving.</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Benefits:</strong> Chia seeds are rich in omega-3 fatty acids, fiber, and protein. This dessert has minimal impact on blood sugar while providing a satisfying sweet treat.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Recipe 2
        st.markdown("<div class='recipe-container'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(r"C:\Users\shail\Downloads\final year project pkl\baked_apple.jpg", use_container_width=True)
        with col2:
            st.markdown("<h3>Cinnamon Baked Apples</h3>", unsafe_allow_html=True)
            st.markdown("<p><strong>Ingredients:</strong> 1 apple, 1 teaspoon cinnamon, 1 tablespoon chopped walnuts, 1 teaspoon honey (optional), 1/4 cup water</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Instructions:</strong> Core the apple and place in a baking dish. Mix cinnamon, walnuts, and honey, then fill the apple core with the mixture. Add water to the dish. Bake at 350°F for 30 minutes until tender.</p>", unsafe_allow_html=True)
            st.markdown("<p><strong>Benefits:</strong> Apples are rich in fiber, particularly pectin, which helps regulate blood sugar and cholesterol. The cinnamon may help improve insulin sensitivity.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Nutritional Tips
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>General Nutritional Guidelines</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='medium-font'>When planning meals for diabetes and heart health, consider these principles:</p>
    <ul>
        <li>Focus on complex carbohydrates with low glycemic index (whole grains, legumes)</li>
        <li>Include plenty of fiber from vegetables, fruits, and whole grains</li>
        <li>Choose lean proteins and plant-based protein sources</li>
        <li>Include heart-healthy fats from sources like olive oil, avocados, nuts, and fatty fish</li>
        <li>Limit added sugars, refined carbohydrates, and processed foods</li>
        <li>Control portion sizes to maintain healthy weight</li>
        <li>Reduce sodium intake to support healthy blood pressure</li>
        <li>Stay well-hydrated, primarily with water</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Meal Planning Tips
    st.markdown("<div class='content-card'>", unsafe_allow_html=True)
    st.markdown("<h2 class='card-title'>Meal Planning Tips</h2>", unsafe_allow_html=True)
    st.markdown("""
    <p class='medium-font'>Consider these strategies when planning your meals:</p>
    <ul>
        <li><strong>The Plate Method:</strong> Fill half your plate with non-starchy vegetables, one quarter with lean protein, and one quarter with whole grains or starchy vegetables</li>
        <li><strong>Regular Eating Schedule:</strong> Eating at consistent times helps maintain stable blood sugar levels</li>
        <li><strong>Balanced Macronutrients:</strong> Include protein, healthy fats, and complex carbohydrates at each meal</li>
        <li><strong>Mindful Eating:</strong> Pay attention to hunger and fullness cues, eat slowly, and savor your food</li>
        <li><strong>Prep Ahead:</strong> Prepare healthy foods in advance to avoid unhealthy choices when pressed for time</li>
    </ul>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Add footer
st.markdown("""
<div style='text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid var(--border-color); color: var(--light-text);'>
    <p>LifeRisk Monitor © 2025 | Developed as a Final Year Project</p>
    <p>Disclaimer: This application is for educational purposes only and should not replace professional medical advice.</p>
</div>
""", unsafe_allow_html=True)