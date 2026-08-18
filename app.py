import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

st.title("✏️ Handwritten Digit Recognizer")
st.write("Draw a digit below and watch the AI recognize it!")

# --- Check if model exists, if not train it ---
@st.cache_resource
def load_or_train_model():
    model_path = 'digit_recognizer.keras'
    
    if os.path.exists(model_path):
        st.write("✅ Loading existing model...")
        return keras.models.load_model(model_path)
    else:
        st.write("⏳ Training model on MNIST data (this will take ~1-2 minutes)...")
        
        # Load MNIST data
        (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
        x_train = x_train / 255.0
        
        # Build model
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])
        
        # Compile and train
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, verbose=1)
        
        # Save the model
        model.save(model_path)
        st.write("✅ Model trained and saved!")
        return model

# Load or train the model
model = load_or_train_model()

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
        # Convert to grayscale
        if canvas_result.image_data.shape[2] == 4:
            rgb = canvas_result.image_data[:, :, :3]
        else:
            rgb = canvas_result.image_data
        
        gray = np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])
        
        # Resize
        img_pil = Image.fromarray(gray.astype('uint8'))
        img_pil = img_pil.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(img_pil) / 255.0
        
        # Predict
        prediction = model.predict(img_array.reshape(1, 28, 28))
        predicted_digit = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # Show results
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(img_array.reshape(28, 28), width=150)
        with col2:
            st.subheader(f"Prediction: {predicted_digit}")
            st.write(f"Confidence: {confidence:.1f}%")
        
        st.bar_chart(prediction[0])
    else:
        st.info("draw a number!")
else:
    st.info("draw a number!")