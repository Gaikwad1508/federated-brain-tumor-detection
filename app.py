import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
# Since app.py is now in root, we import from the src folder
from src.model_architecture import get_model

# --- PAGE CONFIG ---
st.set_page_config(page_title="Brain Tumor AI", page_icon="🧠")

# --- LOAD CHAMPION MODEL ---
@st.cache_resource
def load_tumor_model():
    model = get_model("efficientnet")
    # 🔄 UPDATED PATH: Looking inside your new 'models' folder
    model.load_weights("efficientnet_best.h5")
    return model

model = load_tumor_model()
classes = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']

# --- UI ---
st.title("🧠 Federated Brain Tumor Classifier")
st.markdown("---")

uploaded_file = st.file_uploader("Upload MRI Scan (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Target MRI Scan', use_column_width=True)
    
    # Preprocess
    img_array = np.array(image.convert('RGB'))
    img_resized = cv2.resize(img_array, (224, 224))
    img_normalized = img_resized.astype('float32') / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    if st.button('🚀 Run Diagnostic AI'):
        prediction = model.predict(img_batch)
        # Use Softmax to get clear percentages
        probabilities = tf.nn.softmax(prediction[0])
        conf = np.max(probabilities) * 100
        label = classes[np.argmax(prediction)]

        st.success(f"**Diagnosis:** {label}")
        st.info(f"**AI Confidence:** {conf:.2f}%")
        st.warning("🚨 Disclaimer: Educational tool only. Consult a medical professional.")