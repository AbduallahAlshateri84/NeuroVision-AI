from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Brain MRI Classification",
    page_icon="🧠",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model_exported.keras"

IMAGE_SIZE = (224, 224)

# Must match the exact class order used during training.
CLASS_NAMES = [
    "Glioma",
    "Meningioma",
    "No Tumour",
    "Pituitary Tumour",
]


# =========================================================
# PAGE STYLING
# =========================================================

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(38, 166, 154, 0.12), transparent 30%),
                linear-gradient(135deg, #f7fbfd 0%, #eef5f8 100%);
        }

        .main-title {
            font-size: 2.8rem;
            font-weight: 750;
            color: #102a43;
            margin-bottom: 0.25rem;
        }

        .subtitle {
            color: #486581;
            font-size: 1.08rem;
            margin-bottom: 1.5rem;
        }

        .prediction-card {
            background: rgba(255, 255, 255, 0.95);
            border: 1px solid #c9dbe5;
            border-radius: 18px;
            padding: 24px;
            box-shadow: 0 10px 28px rgba(16, 42, 67, 0.08);
        }

        .prediction-label {
            color: #627d98;
            font-size: 0.95rem;
            margin-bottom: 0.4rem;
        }

        .prediction-value {
            color: #007c91;
            font-size: 2rem;
            font-weight: 750;
            margin-bottom: 0.8rem;
        }

        .confidence-value {
            color: #102a43;
            font-size: 1.2rem;
            font-weight: 650;
        }

        .warning-box {
            background: #fff8e6;
            color: #6b4f00;
            border-left: 5px solid #d69e2e;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 1.2rem;
        }

        .info-box {
            background: #eaf7fa;
            color: #14535d;
            border-left: 5px solid #2c8f9f;
            padding: 15px;
            border-radius: 10px;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource
def load_model() -> tf.keras.Model:
    """
    Load the trained Keras model once and cache it.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found at:\n{MODEL_PATH}\n\n"
            "Place best_model.keras in the same folder as app.py."
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
    )

    return model


# =========================================================
# IMAGE PREPROCESSING
# =========================================================

def prepare_image(
    image: Image.Image,
) -> tuple[np.ndarray, Image.Image]:
    """
    Convert the image to RGB, resize it to 224x224 and apply
    MobileNetV2 preprocessing.

    Returns:
        processed_batch:
            NumPy array with shape (1, 224, 224, 3)

        resized_image:
            PIL image resized to 224x224
    """

    rgb_image = image.convert("RGB")

    resized_image = rgb_image.resize(
        IMAGE_SIZE,
        resample=Image.Resampling.BILINEAR,
    )

    image_array = np.asarray(
        resized_image,
        dtype=np.float32,
    )

    batch = np.expand_dims(
        image_array,
        axis=0,
    )

    processed_batch = preprocess_input(
        batch
    )

    return processed_batch, resized_image


# =========================================================
# PREDICTION
# =========================================================

def predict_image(
    model: tf.keras.Model,
    processed_batch: np.ndarray,
) -> tuple[np.ndarray, int, str, float]:
    """
    Run model inference.

    Returns:
        predictions:
            Probability array for all classes

        predicted_index:
            Index of the highest probability

        predicted_class:
            Human-readable class label

        confidence:
            Highest probability
    """

    predictions = model.predict(
        processed_batch,
        verbose=0,
    )[0]

    predicted_index = int(
        np.argmax(predictions)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        predictions[predicted_index]
    )

    return (
        predictions,
        predicted_index,
        predicted_class,
        confidence,
    )


# =========================================================
# GRAD-CAM HELPERS
# =========================================================

def find_backbone_model(
    model: tf.keras.Model,
) -> tf.keras.Model:
    """
    Find the nested MobileNetV2 model inside the complete model.
    """

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            if "mobilenet" in layer.name.lower():
                return layer

    raise ValueError(
        "Could not find a nested MobileNetV2 model."
    )


def find_last_conv_layer(
    backbone: tf.keras.Model,
) -> tf.keras.layers.Layer:
    """
    Find the final layer in the backbone that produces a 4D feature map.
    """

    for layer in reversed(backbone.layers):
        try:
            output_shape = layer.output.shape

            if len(output_shape) == 4:
                return layer

        except (
            AttributeError,
            TypeError,
            ValueError,
        ):
            continue

    raise ValueError(
        "No suitable convolutional layer was found."
    )


def get_classifier_layers(
    model: tf.keras.Model,
    backbone: tf.keras.Model,
) -> list[tf.keras.layers.Layer]:
    """
    Return the classification layers located after the MobileNetV2 backbone.

    Expected structure:
    MobileNetV2 -> GlobalAveragePooling2D -> Dropout -> Dense
    """

    backbone_index = model.layers.index(
        backbone
    )

    classifier_layers = model.layers[
        backbone_index + 1:
    ]

    if not classifier_layers:
        raise ValueError(
            "No classifier layers were found after the backbone."
        )

    return classifier_layers


# =========================================================
# GRAD-CAM GENERATION
# =========================================================

def generate_gradcam_heatmap(
    model: tf.keras.Model,
    processed_batch: np.ndarray,
    predicted_index: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a normalized Grad-CAM heatmap for the selected class.
    """

    backbone = find_backbone_model(
        model
    )

    last_conv_layer = find_last_conv_layer(
        backbone
    )

    classifier_layers = get_classifier_layers(
        model,
        backbone,
    )

    backbone_grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            last_conv_layer.output,
            backbone.output,
        ],
    )

    input_tensor = tf.convert_to_tensor(
        processed_batch,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:
        conv_outputs, backbone_output = backbone_grad_model(
            input_tensor,
            training=False,
        )

        tape.watch(
            conv_outputs
        )

        predictions = backbone_output

        for layer in classifier_layers:
            predictions = layer(
                predictions,
                training=False,
            )

        if predicted_index is None:
            predicted_index = int(
                tf.argmax(predictions[0])
            )

        target_score = predictions[
            :,
            predicted_index,
        ]

    gradients = tape.gradient(
        target_score,
        conv_outputs,
    )

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradient calculation failed."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    conv_outputs = conv_outputs[0]

    weighted_feature_maps = (
        conv_outputs * pooled_gradients
    )

    heatmap = tf.reduce_sum(
        weighted_feature_maps,
        axis=-1,
    )

    heatmap = tf.nn.relu(
        heatmap
    )

    max_value = tf.reduce_max(
        heatmap
    )

    if float(max_value) > 0:
        heatmap = heatmap / max_value

    return heatmap.numpy()


# =========================================================
# GRAD-CAM OVERLAY
# =========================================================

def create_gradcam_overlay(
    original_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Convert the heatmap into a colored overlay and blend it with
    the original image.
    """

    original_rgb = original_image.convert(
        "RGB"
    )

    original_array = np.asarray(
        original_rgb,
        dtype=np.float32,
    )

    heatmap_uint8 = np.uint8(
        np.clip(
            heatmap,
            0,
            1,
        ) * 255
    )

    heatmap_image = Image.fromarray(
        heatmap_uint8
    )

    heatmap_image = heatmap_image.resize(
        original_rgb.size,
        resample=Image.Resampling.BILINEAR,
    )

    resized_heatmap = np.asarray(
        heatmap_image,
        dtype=np.float32,
    ) / 255.0

    colour_map = matplotlib.colormaps[
        "turbo"
    ]

    coloured_heatmap = colour_map(
        resized_heatmap
    )[:, :, :3]

    coloured_heatmap = (
        coloured_heatmap * 255.0
    )

    blended_image = (
        (1.0 - alpha) * original_array
        + alpha * coloured_heatmap
    )

    blended_image = np.clip(
        blended_image,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(
        blended_image
    )


# =========================================================
# PAGE HEADER
# =========================================================

st.markdown(
    """
    <div class="main-title">
        Brain MRI Classification System
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Upload a brain MRI image to classify it as glioma,
        meningioma, pituitary tumour, or no tumour.
        The Grad-CAM view highlights the image regions that
        most influenced the CNN prediction.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="warning-box">
        <strong>Academic demonstration only:</strong>
        This application is not a medical diagnostic system.
        Do not use its predictions for healthcare or treatment decisions.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# LOAD MODEL
# =========================================================

try:
    model = load_model()

    st.success(
        "CNN model loaded successfully."
    )

except Exception as error:
    st.error(
        f"Unable to load the Keras model: {error}"
    )

    st.stop()


# =========================================================
# FILE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
    accept_multiple_files=False,
    help="Upload a JPG, JPEG, or PNG brain MRI image.",
)


# =========================================================
# IMAGE PROCESSING
# =========================================================

if uploaded_file is not None:
    try:
        uploaded_image = Image.open(
            uploaded_file
        ).convert("RGB")

        processed_batch, resized_image = prepare_image(
            uploaded_image
        )

        with st.spinner(
            "Running CNN prediction and generating Grad-CAM..."
        ):
            (
                predictions,
                predicted_index,
                predicted_class,
                confidence,
            ) = predict_image(
                model,
                processed_batch,
            )

            heatmap = generate_gradcam_heatmap(
                model=model,
                processed_batch=processed_batch,
                predicted_index=predicted_index,
            )

            gradcam_overlay = create_gradcam_overlay(
                original_image=uploaded_image,
                heatmap=heatmap,
                alpha=0.45,
            )

        st.divider()

        st.subheader(
            "MRI analysis"
        )

        original_column, heatmap_column = st.columns(
            2,
            gap="large",
        )

        with original_column:
            st.markdown(
                "#### Original MRI"
            )

            st.image(
                uploaded_image,
                caption=uploaded_file.name,
                width="stretch",
            )

        with heatmap_column:
            st.markdown(
                "#### Grad-CAM heatmap"
            )

            st.image(
                gradcam_overlay,
                caption=(
                    "Red and yellow regions had the greatest "
                    "influence on the CNN prediction."
                ),
                width="stretch",
            )

        st.divider()

        result_column, probability_column = st.columns(
            [0.9, 1.1],
            gap="large",
        )

        with result_column:
            st.markdown(
                "### Prediction result"
            )

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">
                        Predicted class
                    </div>

                    <div class="prediction-value">
                        {predicted_class}
                    </div>

                    <div class="prediction-label">
                        Confidence
                    </div>

                    <div class="confidence-value">
                        {confidence * 100:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with probability_column:
            st.markdown(
                "### Class probabilities"
            )

            probability_data = {
                class_name: float(probability)
                for class_name, probability in zip(
                    CLASS_NAMES,
                    predictions,
                )
            }

            sorted_probabilities = sorted(
                probability_data.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            for (
                class_name,
                probability,
            ) in sorted_probabilities:
                st.write(
                    f"**{class_name}: "
                    f"{probability * 100:.2f}%**"
                )

                st.progress(
                    float(probability)
                )

        st.markdown(
            """
            <div class="info-box">
                Grad-CAM shows which pixels most influenced the model's
                selected class. It does not prove that the highlighted
                region is medically abnormal or clinically meaningful.
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as error:
        st.error(
            "The uploaded image could not be processed."
        )

        st.exception(
            error
        )

else:
    st.info(
        "Upload a brain MRI image to begin classification."
    )
