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

# --- Initialize session state for drawing counter ---
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

# --- FUNNY POP-UP ---
if st.session_state.show_popup:
    # Overlay the entire screen with a dark background
    st.markdown("""
        <style>
        .popup-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.85);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 9999;
        }
        .popup-content {
            background-color: #1e1e2f;
            padding: 60px 80px;
            border-radius: 20px;
            text-align: center;
            max-width: 600px;
            border: 3px solid #ff6b6b;
            box-shadow: 0 0 60px rgba(255, 107, 107, 0.3);
        }
        .popup-content h1 {
            font-size: 48px;
            color: #ff6b6b;
            margin-bottom: 20px;
        }
        .popup-content p {
            font-size: 24px;
            color: #ffffff;
            margin-bottom: 30px;
        }
        .popup-content .stButton button {
            background-color: #ff6b6b;
            color: white;
            font-size: 28px;
            font-weight: bold;
            padding: 20px 60px;
            border-radius: 50px;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .popup-content .stButton button:hover {
            background-color: #ff4757;
            transform: scale(1.05);
            box-shadow: 0 0 40px rgba(255, 107, 107, 0.5);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Render the pop-up using Streamlit components
    with st.container():
        st.markdown('<div class="popup-overlay">', unsafe_allow_html=True)
        st.markdown('<div class="popup-content">', unsafe_allow_html=True)
        
        st.markdown("## ARE YOU NOT ENTERTAINED?")
        st.markdown("### (You've drawn 3 numbers... that's enough for now)")
        
        if st.button("YES", key="entertained_button", use_container_width=True):
            st.session_state.draw_count = 0
            st.session_state.show_popup = False
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

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
        
        # --- Increment counter (only if prediction is decent) ---
        if confidence > 30:  # Only count if it's a real attempt
            st.session_state.draw_count += 1
            if st.session_state.draw_count >= 3:
                st.session_state.show_popup = True
                st.rerun()
        
        # --- Show results ---
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img_array.reshape(28, 28), caption="What the model sees", width=150)
        with col2:
            st.subheader(f"Prediction: {predicted_digit}")
            st.write(f"Confidence: {confidence:.1f}%")
            st.write(f"Draws so far: {st.session_state.draw_count}/3")
        
        st.bar_chart(prediction[0])
    else:
        st.info("ARE YOU NOT ENTERTAINED?")
else:
    st.info("ARE YOU NOT ENTERTAINED?")