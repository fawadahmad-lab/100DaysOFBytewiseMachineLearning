import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('model.h5')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
IMAGE_SIZE = 255

# Function to preprocess and predict
def predict(img):
    # Resize the image to the required size (255, 255)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Convert the image to an array
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    # Normalize the image (if the model expects normalized images)
    img_array /= 255.0

    # Make predictions
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# Streamlit UI
st.title('Potato Disease Classification')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    # Predict the image class and confidence
    predicted_class, confidence = predict(img)
    
    # Display the predictions
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence}%")
