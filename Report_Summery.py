 # Step 1: Setting Up Environment and Installing Required Libraries

# Install necessary libraries

# Step 2: Import Required Libraries
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import pdfplumber
import fitz  # PyMuPDF for image extraction
import re
import google.generativeai as genai
from PIL import Image
import pytesseract


import pytesseract


# Configure Gemini Model
api_key = "Key"  # Replace with your API key
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-pro')

# Step 3: Data Preprocessing for Reports and Prescriptions

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_image(image_file):
    """Extract text from uploaded image file."""
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

def extract_values_from_text(text):
    """Parse report text and extract key metrics."""
    metrics = {}
    pattern = r'([a-zA-Z ]+):\s*([0-9.]+)'  # Example: "Hemoglobin: 13.5"
    
    for match in re.findall(pattern, text):
        try:
            # Only try to convert the value to float if it is numeric
            value = match[1]
            if value.replace('.', '', 1).isdigit():  # Check if the value is numeric
                metrics[match[0].strip()] = float(value)
            else:
                metrics[match[0].strip()] = value  # Keep non-numeric values as they are
        except ValueError:
            continue  # Skip values that cannot be converted to float
    return metrics

# Step 4: Gemini Model Analysis

def analyze_report_with_gemini(text):
    """Analyze report text using Gemini model."""
    # response = model.generate_content(f"Analyze the following medical report text and identify potential diseases and precautions.and suggest for new report if report is older than 3 months and also give near by labotary from the web give link of location and First give highlight of values which value is below and above of thereshold given in report in simple term.: {text}")
    response = model.generate_content(f"""
Analyze the following medical report text very accuratly do not miss any minor details and threshold values take from the report:

1. Highlight the key health metrics in the report and indicate any values that are above or below the provided thresholds. Please provide simple explanations of what these values mean in layman's terms. For example, if a value is above a threshold, what does that imply about the patient's health, and if it is below the threshold, what concerns should be considered?

2. Based on the analyzed metrics, identify any potential diseases or conditions that may be suggested by these results. Include a brief explanation of possible causes and recommendations for further examination or treatment.

3. If the report is older than 3 months, suggest that the patient should get a new report for updated results and re-evaluation of their condition. Provide recommendations on how often such reports should be updated.

4. Finally, if needed, provide information on nearby laboratories where the patient can get further tests done. Include a link to the location and contact details of a local laboratory or diagnostic center for convenience.
                                      {text}
""")


    return response.text

def analyze_prescription_with_gemini(text):
    """Analyze prescription text using Gemini model."""
    response = model.generate_content(f"Extract medicines, their purposes, and a summary from the following prescription text and posible diseases from which patient suffering.")
    return response.text

# Step 5: Streamlit UI for Input and Analysis

def main():
    st.title('AI-Based Medical Analysis System')
    st.write("Upload your medical report or prescription for analysis.")

    # Input type selection
    input_type = st.radio("Select Input Type:", ('Report', 'Prescription'))

    uploaded_file = st.file_uploader("Choose a PDF or Image file", type=["pdf", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = extract_text_from_image(uploaded_file)

        st.text_area("Extracted Text", text, height=300)

        if input_type == 'Report':
            # Extract values from text
            metrics = extract_values_from_text(text)
            st.write("Extracted Metrics:", metrics)

            # Analyze with Gemini Model
            analysis = analyze_report_with_gemini(text)
            st.write("Analysis Result:")
            st.write(analysis)
        else:
            # Analyze prescription text
            analysis = analyze_prescription_with_gemini(text)
            st.write("Prescription Analysis:")
            st.write(analysis)

if __name__ == '__main__':
    main()
