import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page Configuration
st.set_page_config(page_title="Protein Category Predictor", layout="centered")

st.title("🧬 Protein Structural Category Predictor")
st.write("This web application uses Machine Learning to predict a protein's structural domain classification based on its physical and chemical properties.")

# Load Saved Assets
@st.cache_resource
def load_assets():
    model = joblib.load('best_protein_model.pkl')
    scaler = joblib.load('protein_scaler.pkl')
    features = joblib.load('selected_features.pkl')
    return model, scaler, features

try:
    model, scaler, features = load_assets()
    st.success("Machine Learning assets loaded successfully!")
except Exception as e:
    st.error("Error loading model files. Ensure 'best_protein_model.pkl', 'protein_scaler.pkl', and 'selected_features.pkl' are in the same folder.")
    st.stop()

st.subheader("Input Protein Features")
st.write(f"The model requires inputs for these selected properties: {features}")

# Dynamically generate inputs for whatever columns the Filter Method selected
user_inputs = {}
for feat in features:
    default_val = 0.0
    if feat == 'seq_length': default_val = 350.0
    elif feat == 'mol_weight': default_val = 40000.0
    elif feat == 'pI': default_val = 7.0
    elif feat == 'gravy': default_val = -0.4
    
    user_inputs[feat] = st.number_input(f"Enter value for {feat}:", value=float(default_val))

# Predict Button
if st.button("Predict Classification"):
    input_df = pd.DataFrame([user_inputs])
    scaled_input = scaler.transform(input_df)
    
    prediction = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]
    
    st.markdown("---")
    st.subheader("Analysis Result")
    if prediction == 1:
        st.warning("Prediction: **Cellular Component / Membrane Structural Protein**")
    else:
        st.info("Prediction: **Soluble / Other Functional Protein**")
        
    st.write(f"Confidence: {probabilities[prediction]*100:.2f}%")
