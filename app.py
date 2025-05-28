import streamlit as st
import joblib
import numpy as np

# Load the trained model and features
model = joblib.load("xgboost_lung_survival_model.pkl")
feature_names = joblib.load("xgb_feature_columns.pkl")

st.title("🫁 Lung Cancer Survival Prediction")

st.markdown("### Enter patient details:")

# Basic inputs
age = st.slider("Age", 0, 100, 60)
gender = st.selectbox("Gender", ["Male", "Female"])
family_history = st.selectbox("Family History of Cancer?", ["No", "Yes"])
bmi = st.number_input("BMI", value=22.5)
cholesterol = st.number_input("Cholesterol Level", value=180)
hypertension = st.selectbox("Has Hypertension?", ["No", "Yes"])
asthma = st.selectbox("Has Asthma?", ["No", "Yes"])
cirrhosis = st.selectbox("Has Cirrhosis?", ["No", "Yes"])
other_cancer = st.selectbox("Has Other Cancer?", ["No", "Yes"])
treatment_duration = st.slider("Treatment Duration (days)", 0, 1000, 120)

# One-hot options
country = st.selectbox("Country", [f.split("_")[1] for f in feature_names if f.startswith("country_")])
smoking = st.selectbox("Smoking Status", ["Former Smoker", "Never Smoked", "Passive Smoker"])
treatment = st.selectbox("Treatment Type", ["Combined", "Radiation", "Surgery"])
stage = st.selectbox("Cancer Stage", ["Stage II", "Stage III", "Stage IV"])

# Prepare feature vector
def build_features():
    base = [
        age,
        1 if gender == "Female" else 0,
        1 if family_history == "Yes" else 0,
        bmi,
        cholesterol,
        1 if hypertension == "Yes" else 0,
        1 if asthma == "Yes" else 0,
        1 if cirrhosis == "Yes" else 0,
        1 if other_cancer == "Yes" else 0,
        treatment_duration
    ]

    encoded = []
    for fname in feature_names[10:]:  # skip first 10 base columns
        if fname.startswith("country_"):
            encoded.append(1 if fname.endswith(country) else 0)
        elif fname.startswith("smoking_status_"):
            encoded.append(1 if fname.endswith(smoking) else 0)
        elif fname.startswith("treatment_type_"):
            encoded.append(1 if fname.endswith(treatment) else 0)
        elif fname.startswith("cancer_stage_"):
            encoded.append(1 if fname.endswith(stage) else 0)

    return np.array(base + encoded).reshape(1, -1)

# Prediction
if st.button("Predict"):
    input_data = build_features()
    prediction = model.predict(input_data)[0]
    st.success("✅ Survived" if prediction == 1 else "❌ Did Not Survive")
