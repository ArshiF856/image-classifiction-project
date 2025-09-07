import streamlit as st
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Streamlit app
st.title("Image Classification App")
st.write("Upload an image to classify it using MobileNetV2.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Make prediction
    preds = model.predict(x)
    predictions = decode_predictions(preds, top=3)[0]

    # Display the results
    st.write("Predictions:")
    for pred in predictions:
        st.write(f"{pred[1]}: {round(pred[2] * 100, 2)}%")
