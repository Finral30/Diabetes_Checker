import streamlit as st
import pickle
import numpy as np

# Load saved model and scaler
with open('diabetes_model.pkl', 'rb') as file:
    classifier = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# App title
st.title("Diabetes Prediction App 🩺")

st.markdown("""
This app predicts whether a person is likely to have **diabetes** based on health parameters.
""")

# Input fields
pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)
blood_pressure = st.number_input("Blood Pressure value", min_value=0, max_value=200, value=70)
skin_thickness = st.number_input("Skin Thickness value", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=79)
bmi = st.number_input("BMI value", min_value=0.0, max_value=70.0, value=25.0, format="%.2f")
dpf = st.number_input("Diabetes Pedigree Function value", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
age = st.number_input("Age of the Person", min_value=0, max_value=120, value=33)

# Prediction button
if st.button("Predict"):
    # Convert inputs to numpy array and reshape
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    # Standardize the input
    std_data = scaler.transform(input_data_as_numpy_array)

    # Make prediction
    prediction = classifier.predict(std_data)

    # Display result
    if prediction[0] == 0:
        st.success("The person is **not diabetic** 🙂")
    else:
        st.error("The person is **diabetic** 😟")
