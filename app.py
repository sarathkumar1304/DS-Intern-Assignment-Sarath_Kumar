import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# Title
st.title("🏠 Building Energy Consumption Predictor")
st.markdown("This app uses a trained regression model to predict building energy consumption.")

# Load model
@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

model_path = "models/xgboost_pipeline.pkl"
if not os.path.exists(model_path):
    st.error(f"Model file not found at `{model_path}`. Please train and save your model first.")
    st.stop()

model = load_model(model_path)

# Define input fields
st.header("Enter Building Sensor Data")
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    input_values = []
    features = [
         "lighting_energy", 
        "zone1_temperature", "zone1_humidity", "zone2_temperature", "zone2_humidity",
        "zone3_temperature", "zone3_humidity", "zone4_temperature", "zone4_humidity",
        "zone5_temperature", "zone5_humidity", "zone6_temperature", "zone6_humidity",
        "zone7_temperature", "zone7_humidity", "zone8_temperature", "zone8_humidity",
        "zone9_temperature", "zone9_humidity", "outdoor_temperature", "atmospheric_pressure",
        "outdoor_humidity", "wind_speed", "visibility_index", "dew_point", 
        "random_variable1", "random_variable2", "hour", "dayofweek", "month", "is_weekend"
    ]

    for i, feature in enumerate(features):
        if i % 3 == 0:
            with col1:
                val = st.number_input(f"{feature}", value=0.0)
        elif i % 3 == 1:
            with col2:
                val = st.number_input(f"{feature}", value=0.0)
        else:
            with col3:
                val = st.number_input(f"{feature}", value=0.0)
        input_values.append(val)

    submitted = st.form_submit_button("Predict")

# Predict
if submitted:
    input_array = np.array(input_values).reshape(1, -1)
    prediction = model.predict(input_array)[0]
    st.success(f"📊 Predicted Energy Consumption: **{prediction:.2f} units**")
    # st.balloons()