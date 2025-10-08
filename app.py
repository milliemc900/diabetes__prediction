# app.py - Diabetes Prediction App (Random Forest)

import streamlit as st
import pandas as pd
import joblib
import os

# --- Load trained Random Forest model safely ---
@st.cache_resource
def load_model():
    # ✅ Correct relative path
    model_path = os.path.join("models", "RandomForest_model.pkl")

    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}. Please check your repo structure.")
        st.stop()

    # Load the model
    model = joblib.load(model_path)
    return model


# --- Load Model ---
model = load_model()

# --- App Title ---
st.title("🩺 Diabetes Prediction App")

# --- Input Fields ---
st.header("Enter Patient Details:")
Pregnancies = st.number_input("Pregnancies", min_value=0, step=1)
Glucose = st.number_input("Glucose Level", min_value=0)
BloodPressure = st.number_input("Blood Pressure", min_value=0)
SkinThickness = st.number_input("Skin Thickness", min_value=0)
Insulin = st.number_input("Insulin Level", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", min_value=0.0)
Age = st.number_input("Age", min_value=0, step=1)

# --- Predict Button ---
if st.button("Predict Diabetes"):
    input_data = pd.DataFrame(
        [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]],
        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    )

    # --- Make Prediction ---
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("The model predicts that this patient **has diabetes**.")
    else:
        st.success("The model predicts that this patient **does not have diabetes**.")
