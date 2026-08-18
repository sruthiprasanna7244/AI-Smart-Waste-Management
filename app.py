import streamlit as st
import tensorflow as tf
import json
import numpy as np
from PIL import Image

# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="AI Smart Waste Management",
    page_icon="♻️",
    layout="centered"
)

IMAGE_SIZE = (180, 180)

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("waste_model.keras")

model = load_model()

# =========================
# LOAD CLASS NAMES
# =========================

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# =========================
# TITLE
# =========================

st.title("♻️ AI Smart Waste Management")

st.write(
    "Upload a waste image and let AI identify its category."
)

st.divider()

# =========================
# WASTE CATEGORIES
# =========================

st.subheader("🗑️ Supported Waste Categories")

st.write(
    "📦 Cardboard   |   🪟 Glass   |   🔩 Metal"
)

st.write(
    "📄 Paper   |   🧴 Plastic   |   🗑️ Trash"
)

st.divider()

# =========================
# IMAGE UPLOAD
# =========================

st.subheader("📸 Upload Waste Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# PREDICTION
# =========================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image"
    )

    if st.button("🔍 Predict Waste Type"):

        image_resized = image.resize(IMAGE_SIZE)

        img_array = np.array(
            image_resized,
            dtype=np.float32
        )

        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        predictions = model.predict(
            img_array,
            verbose=0
        )

        predicted_index = np.argmax(predictions[0])

        predicted_class = class_names[predicted_index]

        confidence = (
            predictions[0][predicted_index] * 100
        )

        # =========================
        # RESULT
        # =========================

        st.divider()

        st.subheader("🤖 AI Prediction")

        st.success(
            f"♻️ Waste Type: {predicted_class.title()}"
        )

        st.info(
            f"🎯 Confidence: {confidence:.2f}%"
        )

        # =========================
        # DISPOSAL SUGGESTION
        # =========================

        suggestions = {
            "cardboard":
                "📦 Send cardboard for recycling.",

            "glass":
                "🪟 Put glass in a glass recycling collection.",

            "metal":
                "🔩 Send metal items for metal recycling.",

            "paper":
                "📄 Put clean paper in paper recycling.",

            "plastic":
                "🧴 Put suitable plastic in plastic recycling.",

            "trash":
                "🗑️ Dispose of general trash in the appropriate waste bin."
        }

        suggestion = suggestions.get(
            predicted_class.lower(),
            "♻️ Dispose of this waste appropriately."
        )

        st.warning(
            f"💡 Suggestion: {suggestion}"
        )

        # =========================
        # ECO IMPACT
        # =========================

        eco_messages = {
            "cardboard":
                "🌱 Recycling cardboard helps reduce paper waste.",

            "glass":
                "♻️ Glass can often be recycled and reused.",

            "metal":
                "🔩 Recycling metal helps save natural resources.",

            "paper":
                "📄 Recycling paper helps reduce the need for new paper.",

            "plastic":
                "🌍 Proper plastic recycling helps reduce environmental pollution.",

            "trash":
                "🗑️ Dispose of general waste in the appropriate waste bin."
        }

        eco_message = eco_messages.get(
            predicted_class.lower(),
            "🌱 Proper waste segregation helps protect our environment."
        )

        st.success(
            f"🌱 Eco Impact: {eco_message}"
        )

else:

    st.info(
        "👆 Upload a waste image to start prediction."
    )
