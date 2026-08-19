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

# --- Initialize session state ---
if 'draw_count' not in st.session_state:
    st.session_state.draw_count = 0
if 'show_popup' not in st.session_state:
    st.session_state.show_popup = False

# --- Load model ---
@st.cache_resource
def load_model():
    model_path = 'digit_recognizer.keras'
    if os.path.exists(model_path):
        return keras.models.load_model(model_path)
    
    url = "https://huggingface.co/catgat/digits-recognizer/resolve/main/digit_recognizer.keras"
    response = requests.get(url)
    with open(model_path, 'wb') as f:
        f.write(response.content)
    return keras.models.load_model(model_path)

model = load_model()

# --- Center digit function ---
def center_digit(img_array):
    coords = np.argwhere(img_array > 0.05)
    if len(coords) == 0:
        return img_array
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    cropped = img_array[y_min:y_max+1, x_min:x_max+1]
    h, w = cropped.shape
    margin = 2
    if h > 4 and w > 4:
        y_min = max(0, y_min - margin)
        y_max = min(27, y_max + margin)
        x_min = max(0, x_min - margin)
        x_max = min(27, x_max + margin)
        cropped = img_array[y_min:y_max+1, x_min:x_max+1]
        h, w = cropped.shape
    
    target_size = 28
    pad_top = (target_size - h) // 2
    pad_bottom = target_size - h - pad_top
    pad_left = (target_size - w) // 2
    pad_right = target_size - w - pad_left
    
    centered = np.pad(cropped, 
                      ((pad_top, pad_bottom), (pad_left, pad_right)), 
                      mode='constant', 
                      constant_values=0)
    return centered

# --- FUNNY POP-UP (Subtle version using st.dialog) ---
if st.session_state.show_popup:
    @st.dialog("ARE YOU NOT ENTERTAINED?")
    def show_popup():
        st.write("WELL ARE YOU?")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("yes", type="primary", use_container_width=True):
                st.session_state.draw_count = 0
                st.session_state.show_popup = False
                st.rerun()
    
    show_popup()

# --- Main App ---
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
        # --- Process image ---
        if canvas_result.image_data.shape[2] == 4:
            rgb = canvas_result.image_data[:, :, :3]
        else:
            rgb = canvas_result.image_data
        
        gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
        
        img_pil = Image.fromarray(gray.astype('uint8'))
        img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img_pil) / 255.0
        
        img_array = center_digit(img_array)
        
        if img_array.mean() > 0.5:
            img_array = 1.0 - img_array
        
        # --- Predict ---
        prediction = model.predict(img_array.reshape(1, 28, 28))
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # --- Increment counter ---
        if confidence > 30:
            st.session_state.draw_count += 1
            if st.session_state.draw_count >= 6 and not st.session_state.show_popup:
                st.session_state.show_popup = True
                st.rerun()
        
        # --- Show results ---
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img_array.reshape(28, 28), caption="What the model sees", width=150)
        with col2:
            st.subheader(f"Prediction: {predicted_digit}")
            st.write(f"Confidence: {confidence:.1f}%")
            
        
        st.bar_chart(prediction[0])
    else:
        st.info('')
else:
    st.info('')