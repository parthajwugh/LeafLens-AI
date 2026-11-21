
import streamlit as st
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from PIL import Image
st.set_page_config(
    page_title="LeafLens AI | Smart Plant Doctor",
    page_icon="🌿",
    layout="wide"
)

MODEL_PATH = Path("models/best_model.keras")
if not MODEL_PATH.exists():
    MODEL_PATH = Path("best_model.keras")

CLASS_NAMES_PATH = Path("models/class_names.json")
if not CLASS_NAMES_PATH.exists():
    CLASS_NAMES_PATH = Path("class_names.json")


@st.cache_resource
def load_ai_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_ai_model()

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)


print("TOTAL CLASSES:", len(class_names))
print("RICE HEALTHY INDEX:", class_names.index("Rice_Combined__Healthy Leaf"))
print("RICE DISEASE INDEX:", class_names.index("Rice_Combined__Bacterial Leaf Blight"))

def predict_disease(image):
    image = image.convert("RGB")

    image_array = tf.image.resize(
        np.array(image, dtype=np.float32),
        (224, 224)
    ).numpy()

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(
        image_array,
        verbose=0
    )[0]

    print("Prediction Sum:", np.sum(predictions))
    print("Prediction Min:", np.min(predictions))
    print("Prediction Max:", np.max(predictions))

    top_indices = np.argsort(predictions)[::-1][:5]

    print("\n===== TOP 5 PREDICTIONS =====")

    results = []

    for index in top_indices:
        print(
            class_names[index],
            ":",
            round(float(predictions[index] * 100), 2),
            "%"
        )

        results.append({
            "class": class_names[index],
            "confidence": float(predictions[index] * 100)
        })

    return results



st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Playfair+Display:wght@600;700&display=swap');

.stApp {
    background:
        radial-gradient(circle at 10% 10%, #d8f3dc 0%, transparent 32%),
        linear-gradient(135deg, #f5fbf5, #e8f5e9);
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

.hero {
    background: linear-gradient(120deg, #063b2b, #08734d);
    padding: 55px 45px;
    border-radius: 28px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 18px 45px rgba(6,59,43,0.20);
}

.hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 52px;
    color: white;
    margin-bottom: 10px;
}

.hero p {
    color: #d9f5e5;
    font-size: 17px;
    max-width: 650px;
}

.tag {
    color: #b9f6ca;
    font-weight: 700;
    letter-spacing: 3px;
    font-size: 12px;
}

.info-card {
    background: rgba(255,255,255,0.82);
    border: 1px solid #d5ead9;
    padding: 24px;
    border-radius: 20px;
    box-shadow: 0 8px 25px rgba(20,70,40,0.06);
    margin-bottom: 20px;
}

.info-card h3 {
    color: #075b3c;
}

.info-card p {
    color: #476354;
}

div[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #72ad87;
    border-radius: 18px;
    padding: 20px;
}

.stButton button {
    background: linear-gradient(90deg, #08734d, #12a36a);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
    font-weight: 700;
}

.stButton button:hover {
    background: #064d35;
    color: white;
    border: none;
}

.support {
    color: #52715e;
    font-size: 13px;
    text-align: center;
    padding: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="tag">SMART AGRICULTURE • AI TECHNOLOGY</div>
    <h1>LeafLens AI 🌿</h1>
    <p>
        Give your crops a healthier future.
        Upload a leaf image and let Artificial Intelligence
        help identify possible plant diseases.
    </p>
</div>
""", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>🌱 Intelligent Plant Care</h3>
        <p>AI-powered analysis of crop leaf images
        to help recognize possible diseases.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>🌾 8 Supported Crop Groups</h3>
        <p>Corn • Cotton • Potato • Rice • Soybean
        • Sugarcane • Tomato • Wheat</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("## 🔬 Detect Plant Disease")

st.info(
    "🌾 LeafLens AI currently supports only these 8 crop groups: "
    "Corn, Cotton, Potato, Rice, Soybean, Sugarcane, Tomato and Wheat. "
    "Other crops are not supported."
)

st.write("Upload a clear photo of a crop leaf to get started.")

uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear JPG or PNG leaf photograph."
)




if uploaded_file is not None and st.button(
        "🌿 Detect Plant Disease",
        type="primary",
        use_container_width=True
    ):
        with st.spinner("AI is analyzing your leaf image..."):
            image = Image.open(uploaded_file)
            results = predict_disease(image)

        st.markdown("## 🔬 AI Detection Results")

        if results:
            result = results[0]

            if result["confidence"] < 70:
                st.error("🚫 Unsupported Crop or Unclear Image")

                st.warning(
                    "LeafLens AI could not confidently identify "
                    "this image as one of its 8 supported crop groups."
                )

                st.info(
                    "Please upload a clear leaf image of Corn, Cotton, "
                    "Potato, Rice, Soybean, Sugarcane, Tomato or Wheat."
                )

            else:
                crop, disease = result["class"].split("__", 1)

                if "healthy" in disease.lower():
                    st.success("🌿 Great! The leaf appears healthy!")
                    st.markdown("### 🌿 Leaf Status: Healthy")
                else:
                    st.warning(
                        "🦠 Possible disease detected. "
                        "Please inspect the leaf carefully."
                    )
                    st.markdown(
                        f"### 🦠 Possible Disease: {disease}"
                    )

                st.success(f"🌱 Detected Crop: {crop}")

                st.metric(
                    label="AI Confidence",
                    value=f"{result['confidence']:.2f}%"
                )

                st.warning(
                    "Please verify the prediction before taking "
                    "any treatment decision."
                )


# FARMER MASCOT SECTION

st.markdown("---")

st.markdown("""
<div style="text-align:center; padding:20px;">
    <div style="font-size:100px;">👨‍🌾</div>
    <h2>Hello Farmer! 🌿</h2>
    <h3>Let's Grow Healthy Crops Together!</h3>
    <p>
        Your friendly AI farming companion is here
        to help you care for your crops.
    </p>
    <div style="font-size:30px;">🌱 🌾 🍃 🌻</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="support">
    LeafLens AI • Supporting healthier crops through AI 🌿
    <br>
    Predictions are limited to the eight supported crop groups.
    Other crops are not currently supported.
</div>
""", unsafe_allow_html=True)
