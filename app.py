import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas
import os
import requests
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.title("everything's computer 👁️👃👁️")

# --- Load model from Hugging Face ---
@st.cache_resource
def load_model():
    model_path = 'digit_recognizer.keras'
    
    # Check if model exists locally
    if os.path.exists(model_path):
        st.write("✅ Loading existing model...")
        return keras.models.load_model(model_path)
    
    # Download from Hugging Face
    st.write("📥 Downloading model from Hugging Face...")
    
    # YOUR HUGGING FACE MODEL URL
    url = "https://huggingface.co/catgat/digits-recognizer/resolve/main/digit_recognizer.keras"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            f.write(response.content)
        
        st.write("✅ Model downloaded successfully!")
        return keras.models.load_model(model_path)
    
    except Exception as e:
        st.error(f"Error downloading model: {e}")
        st.stop()

# Load the model
model = load_model()

# --- Canvas ---
st.write("Draw a digit below and watch the AI recognize it!")

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
    has_content = np.any(canvas_result.image_data[:, :, :3] > 0)
    
    if has_content:
        # --- IMAGE PROCESSING (Proven working) ---
        
        # Step 1: Get RGB channels (NOT alpha)
        if canvas_result.image_data.shape[2] == 4:
            rgb = canvas_result.image_data[:, :, :3]
        else:
            rgb = canvas_result.image_data
        
        # Step 2: Convert to grayscale using luminance formula
        gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
        
        # Step 3: Resize with high-quality LANCZOS
        img_pil = Image.fromarray(gray.astype('uint8'))
        img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img_pil) / 255.0
        
        # Step 4: Check if image needs inversion (mean pixel value check)
        # MNIST expects black background, white stroke
        if img_array.mean() > 0.5:
            img_array = 1.0 - img_array
        
        # Step 5: Predict
        prediction = model.predict(img_array.reshape(1, 28, 28))
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # Step 6: Show results
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img_array.reshape(28, 28), caption="What the model sees", width=150)
        with col2:
            st.subheader(f"Prediction: {predicted_digit}")
            st.write(f"Confidence: {confidence:.1f}%")
        
        st.bar_chart(prediction[0])
    else:
        st.info("draw a number...and behold!")
else:
    st.info("draw a number...and behold!")