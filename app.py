import streamlit as st
import numpy as np
import joblib

# 1. Load the Bundle
@st.cache_resource
def load_model():
    return joblib.load('phb_model.pkl')

bundle = load_model()
model = bundle['model']
x_scaler = bundle['x_scaler']
y_scaler = bundle['y_scaler']

st.title("PHB Concentration Prediction")
st.write("Enter the nutrient concentrations to predict PHB production.")

# 2. User Inputs (C, N, P)
# We use columns to make it look nice side-by-side
col1, col2, col3 = st.columns(3)

with col1:
    c_conc = st.number_input("C Concentration (g/L)", value=0.0, min_value=0.0)
with col2:
    n_conc = st.number_input("N Concentration (g/L)", value=0.0, min_value=0.0)
with col3:
    p_conc = st.number_input("P Concentration (g/L)", value=0.0, min_value=0.0)

if st.button("Predict PHB"):
    # 3. Prepare Input
    # Must be in the exact order: [C, N, P]
    input_data = np.array([[c_conc, n_conc, p_conc]])
    
    # 4. Scale Input (using x_scaler)
    input_scaled = x_scaler.transform(input_data)
    
    # 5. Predict (Result is still scaled!)
    pred_scaled = model.predict(input_scaled)
    
    # 6. Inverse Scale Output (using y_scaler)
    # We reshape to (-1, 1) because scaler expects a 2D array
    pred_final = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))
    
    # 7. Show Result
    result_value = pred_final[0][0]
    st.success(f"Predicted PHB Concentration: {result_value:.4f} g/L")