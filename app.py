
import streamlit as st
import torch
from PIL import Image
import datetime
import logging
from models import load_model
from preprocessing import preprocess_image, val_transforms
from report_generator import generate_pdf_report

# Configure logger
logging.basicConfig(filename='predictions.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the model and normalization parameters
model, boneage_mean, boneage_std, channel_mean, channel_std = load_model()

# Streamlit UI
st.title("Pediatric Bone Age Prediction")

# Background image (replace with actual URL or path if desired)
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://example.com/clinical-background.jpg");  # Replace with actual URL
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Input fields
st.header("Assessment Details")
tester_name = st.text_input("Tester Name (Radiologist/Doctor/Researcher, required):", key="tester_name")
st.header("Patient Information")
patient_name = st.text_input("Patient Name (optional, e.g., 'Baby of Mrs. Sharma'):", key="patient_name", placeholder="Enter 'Baby of [Parent Name]' if no patient name")
patient_id = st.text_input("Patient ID (optional):", key="patient_id")
gender = st.selectbox("Select Gender:", ["Male", "Female"])
uploaded_file = st.file_uploader("Upload X-ray image:", type=["jpg", "png"])

if st.button("Submit"):
    if uploaded_file is not None and tester_name.strip():
        image = Image.open(uploaded_file)
        preprocessed_img, resized_img = preprocess_image(image)
        
        # Convert to PIL for transform
        pil_img = Image.fromarray(preprocessed_img)
        input_tensor = val_transforms(pil_img).unsqueeze(0)
        
        gender_tensor = torch.tensor([1.0 if gender == 'Male' else 0.0])
        
        with torch.no_grad():
            pred = model(input_tensor, gender_tensor).item()
        predicted_age = pred * boneage_std + boneage_mean
        
        # Log details
        current_time = datetime.datetime.now()
        log_entry = f"Tester: {tester_name}, Date: {current_time}, Patient: {patient_name or 'Anonymous'}, Patient ID: {patient_id or 'Not Provided'}, Gender: {gender}, Predicted Age: {predicted_age:.2f}"
        logging.info(log_entry)
        
        # Display results
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            st.image(resized_img, caption="Preprocessed Image")
        
        st.success(f"Predicted Bone Age: {predicted_age:.2f} months")
        
        # Generate and provide downloadable PDF report
        pdf_bytes = generate_pdf_report(tester_name, patient_name, patient_id, current_time, gender, predicted_age, image, resized_img)
        st.download_button(
            label="Download Report",
            data=pdf_bytes,
            file_name=f"bone_age_report_{tester_name}_{current_time.strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
    else:
        if not tester_name.strip():
            st.error("Please enter the Tester Name.")
        if not uploaded_file:
            st.error("Please upload an X-ray image.")

# Developer footer
st.markdown("<p style='text-align: center; font-size: 12px; color: #666;'>Developed by Ruchi Rathod</p>", unsafe_allow_html=True)