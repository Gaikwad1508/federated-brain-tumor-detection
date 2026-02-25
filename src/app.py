import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import os

# ---------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Federated Tumor Detector",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        color: white;
        background-color: #2ecc71;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-weight: bold;
    }
    .stAlert {
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# TITLE & DESCRIPTION
# ---------------------------------------------------------------------
st.title("🧠 Federated Brain Tumor Detection")
st.markdown("""
This AI diagnostic tool uses a **Federated Learning Model (MobileNetV2)**. 
It was trained collaboratively across multiple hospitals (clients) without ever sharing raw patient data, preserving privacy while achieving high accuracy.
""")
st.write("---")

# ---------------------------------------------------------------------
# MODEL LOADING
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Loads the model once and caches it to speed up the app."""
    # Look for the model in the current directory
    model_path = "best_brain_tumor_model.h5"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Error: Model file '{model_path}' not found in the directory!")
        return None
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# Class Labels (Must match the training order exactly)
CLASSES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# ---------------------------------------------------------------------
# FILE UPLOADER
# ---------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload an MRI Scan (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # 1. Load and Display the Image (Keep it RGB for display)
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Patient MRI', use_column_width=True)
    
    st.write("🔍 **Analyzing Scan...**")
    
    # -----------------------------------------------------------------
    # PREPROCESSING (The Critical Part)
    # -----------------------------------------------------------------
    try:
        # Convert PIL image to NumPy array
        img_array = np.array(image)

        # Handle different image formats to ensure we get 3-channel BGR
        # OpenCV expects BGR, but PIL gives RGB. We MUST convert.
        
        if len(img_array.shape) == 2:  
            # Case 1: Grayscale (2D) -> Convert to BGR
            img_processed = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            
        elif img_array.shape[2] == 4:  
            # Case 2: RGBA (Transparent) -> Convert to BGR
            img_processed = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            
        else:
            # Case 3: Standard RGB -> Convert to BGR (CRITICAL FIX)
            # The model was trained on BGR images (cv2.imread), so we swap colors.
            img_processed = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        # Resize to 224x224 (Model Input Size)
        img_resized = cv2.resize(img_processed, (224, 224))
        
        # Normalize pixel values to 0-1 range
        img_normalized = img_resized.astype('float32') / 255.0
        
        # Add Batch Dimension (1, 224, 224, 3)
        img_batch = np.expand_dims(img_normalized, axis=0)

        # -----------------------------------------------------------------
        # PREDICTION
        # -----------------------------------------------------------------
        prediction = model.predict(img_batch)
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        result = CLASSES[class_idx]

        # -----------------------------------------------------------------
        # DISPLAY RESULTS
        # -----------------------------------------------------------------
        st.write("---")
        
        # Dynamic Color Logic for Results
        if result == "No Tumor":
            st.success(f"✅ **Diagnosis:** {result}")
        else:
            st.error(f"⚠️ **Diagnosis:** {result}")
            
        # Display Confidence Score
        st.info(f"**Confidence Score:** {confidence:.2f}%")
        
        # Visualization: Probability Distribution
        st.write("### Probability Breakdown")
        # Create a dictionary for the bar chart
        prob_data = {CLASSES[i]: float(prediction[0][i]) for i in range(4)}
        st.bar_chart(prob_data)

    except Exception as e:
        st.error(f"⚠️ An error occurred during processing: {e}")