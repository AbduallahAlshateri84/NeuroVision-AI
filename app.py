from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ---------------------------------------------------------
# Application settings
# ---------------------------------------------------------
st.set_page_config(
    page_title="Brain MRI Classification",
    page_icon="🧠",
    layout="wide",
)

MODEL_PATH = Path(__file__).parent / "best_model.keras"

# This order must match the training dataset class index order.
CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumour",
    "Pituitary Tumour",
]

IMAGE_SIZE = (224, 224)


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top right, #e8f7ff 0%, transparent 30%),
                linear-gradient(135deg, #f8fbff 0%, #eef4f8 100%);
        }

        .main-title {
            font-size: 2.7rem;
            font-weight: 750;
            color: #102a43;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #486581;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        .result-card {
            background: rgba(255, 255, 255, 0.92);
            border: 1px solid #cbdbe7;
            border-radius: 16px;
            padding: 22px;
            box-shadow: 0 8px 24px rgba(16, 42, 67, 0.08);
        }

        .prediction {
            font-size: 1.8rem;
            font-weight: 700;
            color: #007c91;
        }

        .warning-card {
            background: #fff8e6;
            border-left: 5px solid #d69e2e;
            padding: 14px;
            border-radius: 8px;
            color: #5f4700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Load model only once
# ---------------------------------------------------------
@st.cache_resource
def load_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found at: {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )


# ---------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------
def prepare_image(image: Image.Image) -> tuple[np.ndarray, Image.Image]:
    rgb_image = image.convert("RGB")
    resized_image = rgb_image.resize(IMAGE_SIZE)

    image_array = np.asarray(resized_image, dtype=np.float32)
    batch = np.expand_dims(image_array, axis=0)

    # MobileNetV2 preprocessing converts pixels to approximately [-1, 1].
    processed_batch = preprocess_input(batch)

    return processed_batch, resized_image


# ---------------------------------------------------------
# Model prediction
# ---------------------------------------------------------
def predict_image(
    model: tf.keras.Model,
    image: Image.Image,
) -> tuple[str, float, dict[str, float]]:
    processed_batch, _ = prepare_image(image)

    predictions = model.predict(processed_batch, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(CLASS_NAMES, predictions)
    }

    return predicted_class, confidence, probabilities


# ---------------------------------------------------------
# Page header
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">Brain MRI Classification System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Upload a brain MRI image to classify it as glioma, meningioma,
        pituitary tumour, or no tumour.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="warning-card">
        <strong>Academic demonstration:</strong>
        This application is not a medical diagnostic system. Predictions
        must not be used for treatment or healthcare decisions.
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")

# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
try:
    model = load_model()
    st.success("CNN model loaded successfully.")
except Exception as error:
    st.error(f"Unable to load the Keras model: {error}")
    st.stop()


# ---------------------------------------------------------
# Upload section
# ---------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=False,
    help="Upload a JPG, JPEG, or PNG MRI image.",
)

if uploaded_file is not None:
    try:
        uploaded_image = Image.open(uploaded_file)

        image_column, result_column = st.columns([1, 1.2], gap="large")

        with image_column:
            st.subheader("Uploaded MRI")
            st.image(
                uploaded_image,
                caption=uploaded_file.name,
                width="stretch",
            )

        with result_column:
            st.subheader("CNN analysis")

            with st.spinner("Processing the MRI image..."):
                predicted_class, confidence, probabilities = predict_image(
                    model,
                    uploaded_image,
                )

            st.markdown(
                f"""
                <div class="result-card">
                    <div>Predicted class</div>
                    <div class="prediction">{predicted_class}</div>
                    <br>
                    <div>
                        Confidence:
                        <strong>{confidence * 100:.2f}%</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")
            st.subheader("Class probabilities")

            sorted_probabilities = sorted(
                probabilities.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            for class_name, probability in sorted_probabilities:
                st.write(
                    f"**{class_name}: {probability * 100:.2f}%**"
                )
                st.progress(float(probability))

    except Exception as error:
        st.error(f"Unable to process this image: {error}")
else:
    st.info("Upload an MRI image to begin classification.")
