%%writefile app.py
# ------------------------------------------------------------
# 🐟 Multiclass Fish Image Classification Dashboard (Streamlit)
# ------------------------------------------------------------

import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------
# Streamlit Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="Fish Classification", page_icon="🐟", layout="wide")

# ------------------------------------------------------------
# Sidebar Navigation
# ------------------------------------------------------------
with st.sidebar:
    selected = option_menu(
        menu_title="Navigation",
        options=["Upload Image", "Classify"],
        icons=["cloud-upload", "search"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#E6E6FA"},
            "icon": {"color": "#FF00FF", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "color": "#333",
                "padding": "10px",
                "border-radius": "8px",
            },
            "nav-link-selected": {"background-color": "#DDA0DD", "color": "white"},
        },
    )

# ------------------------------------------------------------
# UPLOAD IMAGE PAGE
# ------------------------------------------------------------
if selected == "Upload Image":
    st.markdown("<h1 style='color: #C71585;'>📤 Upload an Image for Classification</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Load trained model
        model_path = "/content/mobilenet_fish_final.keras"  # <-- change to your model path
        model = tf.keras.models.load_model(model_path)

        # Predict
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]

        class_labels = [
            "animal fish", "animal fish bass", "fish sea_food black_sea_sprat",
            "fish sea_food gilt_head_bream", "fish sea_food hourse_mackerel",
            "fish sea_food red_mullet", "fish sea_food red_sea_bream",
            "fish sea_food sea_bass", "fish sea_food shrimp",
            "fish sea_food striped_red_mullet", "fish sea_food trout"
        ]

        predicted_label = class_labels[predicted_class]
        confidence = np.max(prediction) * 100

        st.subheader(f"🎯 Predicted Fish Species: **{predicted_label}**")
        st.write(f"🔍 Confidence Score: **{confidence:.2f}%**")

# ------------------------------------------------------------
# CLASSIFY PAGE
# ------------------------------------------------------------
elif selected == "Classify":
    st.markdown("<h1 style='color: #C71585;'>🔍 Classify Fish Species</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a fish image for classification...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img_array = np.array(image.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        model_path = "/content/mobilenet_fish_final.keras"  # <-- change to your model path
        model = tf.keras.models.load_model(model_path)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]

        class_labels = [
            "animal fish", "animal fish bass", "fish sea_food black_sea_sprat",
            "fish sea_food gilt_head_bream", "fish sea_food hourse_mackerel",
            "fish sea_food red_mullet", "fish sea_food red_sea_bream",
            "fish sea_food sea_bass", "fish sea_food shrimp",
            "fish sea_food striped_red_mullet", "fish sea_food trout"
        ]

        predicted_label = class_labels[predicted_class]
        confidence_score = np.max(predictions) * 100

        st.subheader(f"🎯 Predicted Fish Species: **{predicted_label}**")
        st.write(f"🔍 Confidence Score: **{confidence_score:.2f}%**")

        st.subheader("📊 Confidence Scores for All Classes")
        for i, label in enumerate(class_labels):
            st.write(f"**{label}:** {predictions[0][i] * 100:.2f}%")

# ============================================================
# 🚀 Run Streamlit inside Google Colab (any app file you choose)
# ============================================================

# 1️⃣ Install dependencies (only needed once)
!pip install streamlit pyngrok --quiet

# 2️⃣ Import & setup ngrok
import os
from pyngrok import ngrok

# (Optional) Stop any existing tunnels
ngrok.kill()

# (🔒 replace with your own Ngrok auth token from dashboard.ngrok.com)
NGROK_AUTH_TOKEN = "34Q25J6mc3d2R6Kynlaw3J22WJY_7vo3DE64A4FCDAKTcYXPE"

ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# 3️⃣ Environment tweaks so Streamlit behaves in Colab
os.environ["BROWSER"] = "none"
os.environ["STREAMLIT_GATHER_USAGE_STATS"] = "false"

# 4️⃣ Choose a port and start Streamlit in background
PORT = 8508
!streamlit run /content/app.py --server.port {PORT} &>/dev/null &

# 5️⃣ Create a tunnel and show the link
public_url = ngrok.connect(addr=PORT, proto="http")
print(f"🌐 Streamlit App is live at:\n👉 {public_url.public_url}")
