import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load your best model
model = load_model('custom_cnn_model.h5')
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

st.title("Brain Tumor MRI Classifier")

uploaded_file = st.file_uploader("Upload an MRI image (JPG/PNG):", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L').resize((224,224))
    st.image(img, caption='Uploaded MRI Image', use_column_width=True)
    arr = np.array(img) / 255.0
    arr = arr.reshape(1,224,224,1)
    pred = model.predict(arr)
    predicted_class = class_names[np.argmax(pred)]
    confidence = np.max(pred)
    st.write(f"**Predicted Tumor Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")