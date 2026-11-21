
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI image to get a model prediction.")

MODEL_PATH = Path(__file__).parent / "notebooks" / "brain_tumor_model.h5"

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            resized_image = image.resize((224, 224))
            img_array = np.array(resized_image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array, verbose=0)[0]

            classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
            predicted_index = np.argmax(predictions)

            st.subheader("Prediction Result")
            st.write(f"**Predicted Class:** {classes[predicted_index]}")
            st.write(f"**Model Confidence:** {predictions[predicted_index] * 100:.2f}%")

            st.subheader("Class Probabilities")
            for name, probability in zip(classes, predictions):
                st.write(f"{name}: {probability * 100:.2f}%")
                st.progress(float(probability))

st.warning(
    "Educational project only. This model is not a medical diagnostic tool. "
    "Please consult a qualified medical professional for medical advice."
)

