import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import geocoder
import smtplib
from email.mime.text import MIMEText

# Load the trained model
model = tf.keras.models.load_model("fire_detection_model.keras")

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((180, 180))  # Resize to match model input size
    image = np.array(image)  # Convert to NumPy array
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to get live location
def get_location():
    g = geocoder.ip('me')  # Get location using IP address
    if g.ok:
        return f"Latitude: {g.latlng[0]}, Longitude: {g.latlng[1]}"
    return "Location not available"

# Function to send email
def send_email(location, recipient_email="abhinaypyasi@gmail.com"):
    sender_email = "a80614436@gmail.com"
    sender_password = "fulf cqad ktxf nzyy"  # Use App Password for security
    subject = "🔥 Fire Alert with Live Location!"

    message = MIMEText(f"Fire detected! 🚨\nLocation: {location}\nPlease take immediate action!")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = recipient_email

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, recipient_email, message.as_string())
        server.quit()
        return "✅ Email sent successfully!"
    except Exception as e:
        return f"Error sending email: {e}"

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
    
    if fire_probability <= 0.5:
        st.error(f"🔥 Fire Detected! ")

        # Get live location
        location = get_location()
        st.warning(f"📍 Live Location: {location}")

        # Send email alert
        email_status = send_email(location)
        st.info(email_status)
        
    else:
        st.success(f"✅ No Fire Detected")
