import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("fire_detection_model.keras")

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((180, 180))  # Resize to match model input size
    image = np.array(image)  # Convert to NumPy array
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert to BGR (if needed)
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("🔥 Fire Detection App 🔥")
st.write("Upload an image to check if fire is present.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and make prediction
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Interpret the result
    fire_probability = prediction[0][0]
    
    if fire_probability < 0.5:
        st.error(f"🔥 Fire Detected! ")
    else:
        st.success(f"✅ No Fire Detected")
