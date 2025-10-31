import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet50 import preprocess_input


model = tf.keras.models.load_model(r"\waste\waste_classifier.keras")

class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

st.set_page_config(page_title="Waste Classification", page_icon="♻️", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>
        ♻️ Waste Classification AI
    </h1>
    <p style='text-align: center; font-size:18px; color:#555;'>
        Upload an image of waste and let the AI classify it.<br>
        Model: <b>MobileNetV2 </b>
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📷 Upload a waste image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        if st.button("🔍 Classify"):
            with st.spinner("Analyzing image..."):
                
                img_resized = img.resize((180, 180))
                img_array = np.array(img_resized).astype("float32")
                img_array = np.expand_dims(img_array, axis=0)
                img_array = preprocess_input(img_array)


                
                predictions = model.predict(img_array)
                pred_class = np.argmax(predictions[0])
                confidence = np.max(predictions[0])

              
                st.subheader("Prediction")
                st.success(f"**{class_names[pred_class]}** ({confidence:.2%} confidence)")

                st.subheader("Class probabilities")
                fig, ax = plt.subplots()
                ax.bar(class_names, predictions[0], color="#4CAF50")
                ax.set_ylabel("Probability")
                ax.set_ylim([0, 1])
                plt.xticks(rotation=30)
                st.pyplot(fig)
