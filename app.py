import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import StringIO

# Set page config
st.set_page_config(
    page_title="Engine Motor",
    page_icon="⚙️",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load(r"C:\Users\lenovo\Desktop\engine motor\classifier_model.pkl")
    scaler = joblib.load(r"C:\Users\lenovo\Desktop\engine motor\scaler.pkl")
    return model, scaler

model, scaler = load_model()

# App title and description
st.title("⚙️ Equipment ClassID Classifier")
st.markdown("""
This app predicts the ClassID based on sensor readings from equipment components.
""")

# Get expected feature names from the scaler
expected_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else [
    'Channel name', 'Tachometer', 'Motor', 'Bearing 1 Z', 'Bearing 1 Y', 
    'Bearing 1 X', 'Bearing 2 Z', 'Bearing 2 Y', 'Bearing 2 X', 'Gearbox'
]

# Create two tabs
tab1, tab2 = st.tabs(["Motor Engine Classification", "Batch Prediction"])

with tab1:
    st.header("Motor Engine Classification")
    
    # Create input form
    with st.form("single_pred_form"):
        col1, col2, col3 = st.columns(3)
        
        # Check if we need to include Channel name
        include_channel = 'Channel name' in expected_features
        
        if include_channel:
            channel_name = st.text_input("Channel name", value="default_channel")
        
        with col1:
            tachometer = st.number_input("Tachometer", value=-0.86, format="%.5f")
            motor = st.number_input("Motor", value=-0.01, format="%.5f")
            bearing1_z = st.number_input("Bearing 1 Z", value=0.01, format="%.5f")
        
        with col2:
            bearing1_y = st.number_input("Bearing 1 Y", value=-0.00, format="%.5f")
            bearing1_x = st.number_input("Bearing 1 X", value=-0.01, format="%.5f")
            bearing2_z = st.number_input("Bearing 2 Z", value=-0.00, format="%.5f")
        
        with col3:
            bearing2_y = st.number_input("Bearing 2 Y", value=0.00, format="%.5f")
            bearing2_x = st.number_input("Bearing 2 X", value=0.01, format="%.5f")
            gearbox = st.number_input("Gearbox", value=0.01, format="%.5f")
        
        submitted = st.form_submit_button("Predict ClassID")
    
    if submitted:
        # Create input dictionary with all expected features
        input_dict = {
            'Tachometer': [tachometer],
            'Motor': [motor],
            'Bearing 1 Z': [bearing1_z],
            'Bearing 1 Y': [bearing1_y],
            'Bearing 1 X': [bearing1_x],
            'Bearing 2 Z': [bearing2_z],
            'Bearing 2 Y': [bearing2_y],
            'Bearing 2 X': [bearing2_x],
            'Gearbox': [gearbox]
        }
        
        if include_channel:
            input_dict['Channel name'] = [channel_name]
        
        # Create DataFrame with correct column order
        input_data = pd.DataFrame(input_dict)[expected_features]
        
        try:
            # Scale the data
            scaled_data = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(scaled_data)
            proba = model.predict_proba(scaled_data)
            
            # Display results
            st.success(f"Predicted ClassID: {prediction[0]}")
            
            # Show probabilities
            st.subheader("Prediction Probabilities")
            proba_df = pd.DataFrame({
                'ClassID': model.classes_,
                'Probability': proba[0]
            }).sort_values('Probability', ascending=False)
            st.bar_chart(proba_df.set_index('ClassID'))
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.error(f"Expected features: {', '.join(expected_features)}")

with tab2:
    st.header("Batch Prediction")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload CSV file with sensor data",
        type=['csv'],
        help=f"File should contain these columns: {', '.join(expected_features)}"
    )
    
    if uploaded_file is not None:
        # Read the file
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check if all required columns are present
            missing_cols = [col for col in expected_features if col not in df.columns]
            
            if not missing_cols:
                st.success("File successfully uploaded!")
                st.dataframe(df.head())
                
                if st.button("Predict Batch"):
                    try:
                        # Select and order columns correctly
                        df = df[expected_features]
                        
                        # Scale the data
                        scaled_data = scaler.transform(df)
                        
                        # Make predictions
                        predictions = model.predict(scaled_data)
                        probas = model.predict_proba(scaled_data)
                        
                        # Add predictions to dataframe
                        df['Predicted_ClassID'] = predictions
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.dataframe(df)
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Predictions",
                            data=csv,
                            file_name='predictions.csv',
                            mime='text/csv'
                        )
                        
                        # Show class distribution
                        st.subheader("Class Distribution")
                        class_dist = df['Predicted_ClassID'].value_counts().reset_index()
                        class_dist.columns = ['ClassID', 'Count']
                        st.bar_chart(class_dist.set_index('ClassID'))
                    
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
            else:
                st.error(f"Missing columns: {', '.join(missing_cols)}")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# Add sidebar with info
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    This app uses a machine learning model to predict equipment ClassID based on sensor data.
    """)
    
    st.markdown("## Model Info")
    st.text(f"Model type: {type(model).__name__}")
    st.text(f"Expected features ({len(expected_features)}):")
    st.text("\n".join([f"- {feat}" for feat in expected_features]))
    st.text(f"Classes: {model.classes_}")
    
    st.markdown("## How to Use")
    st.markdown("""
    1. For single prediction: Fill the form and click 'Predict'
    2. For batch prediction: Upload a CSV file with all expected columns
    """)