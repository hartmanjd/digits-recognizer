import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Load model
model = keras.models.load_model('digit_recognizer.keras')

st.title("✏️ Handwritten Digit Recognizer")
st.write("Draw a digit below and watch the AI recognize it!")

# Canvas with REAL-TIME updates
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
    update_streamlit=True,
)

if canvas_result.image_data is not None:
    # Check if anything was drawn
    has_content = np.any(canvas_result.image_data[:, :, :3] > 0)
    
    if has_content:
        # --- FIX: Use RGB channels properly ---
        # The canvas returns RGBA or RGB
        # We want to convert to grayscale properly
        
        if canvas_result.image_data.shape[2] == 4:
            # RGBA - use RGB channels, ignore alpha
            rgb = canvas_result.image_data[:, :, :3]
        else:
            # RGB
            rgb = canvas_result.image_data
        
        # Convert RGB to grayscale using luminance formula
        # This gives us white strokes on black background
        gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
        
        # The canvas has black (0) background and white (255) strokes
        # This matches MNIST format (black background, white digit)
        # So we don't need to invert!
        
        # Resize to 28x28
        img_pil = Image.fromarray(gray.astype('uint8'))
        img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img_pil) / 255.0
        
        # Now predict
        prediction = model.predict(img_array.reshape(1, 28, 28))
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # Show the image and result
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img_array.reshape(28, 28), caption="What model sees", width=150)
        with col2:
            st.subheader(f"Prediction: {predicted_digit}")
            st.write(f"Confidence: {confidence:.1f}%")
        
        st.bar_chart(prediction[0])
    else:
        st.info("✏️ Draw a digit on the canvas above!")
else:
    st.info("✏️ Draw a digit on the canvas above!")