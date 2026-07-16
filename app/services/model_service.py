from __future__ import annotations

import base64
import io
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

LOGGER = logging.getLogger(__name__)


DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumour",
    "no_tumor": "No Tumour",
    "pituitary": "Pituitary Tumour",
}


class ModelNotReadyError(RuntimeError):
    """Raised when inference is requested before a model is available."""


class InvalidImageError(ValueError):
    """Raised when the uploaded file cannot be decoded as an image."""


@dataclass
class ModelStatus:
    ready: bool
    model_path: str
    message: str
    input_size: tuple[int, int]
    classes: list[str]


class ModelService:
    """Loads the Keras model once and provides prediction plus Grad-CAM."""

    def __init__(self, model_path: Path, class_names: tuple[str, ...]) -> None:
        self.model_path = model_path
        self.class_names = list(class_names)
        self.input_size = (224, 224)
        self.model: Any | None = None
        self.grad_model: Any | None = None
        self.backbone: Any | None = None
        self.last_conv_name: str | None = None
        self.error_message = "Model has not been loaded."
        self._lock = threading.Lock()
        self._tf: Any | None = None
        self._preprocess_input: Any | None = None

    def load(self) -> None:
        if not self.model_path.exists():
            self.error_message = (
                f"Model file was not found at {self.model_path}. "
                "Copy best_model.keras into the models folder."
            )
            LOGGER.warning(self.error_message)
            return

        try:
            import tensorflow as tf
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

            self._tf = tf
            self._preprocess_input = preprocess_input
            self.model = tf.keras.models.load_model(self.model_path, compile=False)

            model_shape = self.model.input_shape
            if isinstance(model_shape, list):
                model_shape = model_shape[0]
            if len(model_shape) >= 3 and model_shape[1] and model_shape[2]:
                self.input_size = (int(model_shape[2]), int(model_shape[1]))

            output_units = int(self.model.output_shape[-1])
            if output_units != len(self.class_names):
                raise ValueError(
                    "The model output has "
                    f"{output_units} units, but CLASS_NAMES contains "
                    f"{len(self.class_names)} labels."
                )

            self._build_gradcam_model()
            self.error_message = "Model loaded successfully."
            LOGGER.info(
                "Loaded model from %s with classes %s",
                self.model_path,
                self.class_names,
            )
        except Exception as exc:  # startup should keep the UI available
            self.model = None
            self.grad_model = None
            self.error_message = f"Unable to load model: {exc}"
            LOGGER.exception("Model loading failed")

    def _build_gradcam_model(self) -> None:
        assert self.model is not None
        assert self._tf is not None
        tf = self._tf

        nested_models = [
            layer for layer in self.model.layers if isinstance(layer, tf.keras.Model)
        ]
        if not nested_models:
            raise ValueError("No nested CNN backbone was found in the Keras model.")

        self.backbone = next(
            (
                layer
                for layer in nested_models
                if "mobilenet" in layer.name.lower()
            ),
            nested_models[0],
        )

        conv_layers = [
            layer
            for layer in self.backbone.layers
            if isinstance(layer, tf.keras.layers.Conv2D)
        ]
        if not conv_layers:
            raise ValueError("No Conv2D layer was found for Grad-CAM.")
        last_conv = conv_layers[-1]
        self.last_conv_name = last_conv.name

        head_output = self.backbone.output
        backbone_index = self.model.layers.index(self.backbone)
        for layer in self.model.layers[backbone_index + 1 :]:
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            try:
                head_output = layer(head_output, training=False)
            except TypeError:
                head_output = layer(head_output)

        self.grad_model = tf.keras.models.Model(
            self.backbone.input,
            [last_conv.output, head_output],
            name="gradcam_model",
        )

    def status(self) -> ModelStatus:
        return ModelStatus(
            ready=self.model is not None,
            model_path=str(self.model_path),
            message=self.error_message,
            input_size=self.input_size,
            classes=[self._display_name(name) for name in self.class_names],
        )

    def predict(self, raw_bytes: bytes) -> dict[str, Any]:
        if self.model is None or self._tf is None:
            raise ModelNotReadyError(self.error_message)

        original = self._decode_image(raw_bytes)
        model_image = original.resize(self.input_size, Image.Resampling.LANCZOS)
        image_array = np.asarray(model_image, dtype=np.float32)
        batch = np.expand_dims(image_array, axis=0)

        with self._lock:
            probabilities = self.model.predict(batch, verbose=0)[0]
            predicted_index = int(np.argmax(probabilities))
            heatmap = self._make_gradcam(batch, predicted_index)

        overlay_data_url = self._create_overlay_data_url(original, heatmap)
        original_data_url = self._image_to_data_url(original)

        probability_rows = [
            {
                "class_key": class_name,
                "label": self._display_name(class_name),
                "probability": round(float(probabilities[index]), 6),
                "percentage": round(float(probabilities[index]) * 100, 2),
            }
            for index, class_name in enumerate(self.class_names)
        ]
        probability_rows.sort(key=lambda row: row["probability"], reverse=True)

        predicted_key = self.class_names[predicted_index]
        return {
            "prediction": {
                "class_key": predicted_key,
                "label": self._display_name(predicted_key),
                "confidence": round(float(probabilities[predicted_index]), 6),
                "confidence_percentage": round(
                    float(probabilities[predicted_index]) * 100, 2
                ),
            },
            "probabilities": probability_rows,
            "images": {
                "original": original_data_url,
                "gradcam": overlay_data_url,
            },
            "model": {
                "architecture": "MobileNetV2 transfer-learning CNN",
                "input_size": list(self.input_size),
                "last_gradcam_layer": self.last_conv_name,
            },
            "notice": (
                "Research prototype only. This output is not a medical diagnosis "
                "and must be reviewed by a qualified clinician."
            ),
        }

    @staticmethod
    def _decode_image(raw_bytes: bytes) -> Image.Image:
        try:
            with Image.open(io.BytesIO(raw_bytes)) as image:
                image = ImageOps.exif_transpose(image)
                return image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError(
                "The uploaded file is not a readable PNG, JPEG, WEBP, BMP, or TIFF image."
            ) from exc

    def _make_gradcam(self, batch: np.ndarray, predicted_index: int) -> np.ndarray:
        assert self.grad_model is not None
        assert self._tf is not None
        assert self._preprocess_input is not None
        tf = self._tf

        preprocessed = self._preprocess_input(tf.cast(batch, tf.float32))
        with tf.GradientTape() as tape:
            conv_output, predictions = self.grad_model(preprocessed, training=False)
            class_score = predictions[:, predicted_index]

        gradients = tape.gradient(class_score, conv_output)
        if gradients is None:
            raise RuntimeError("Grad-CAM gradients could not be calculated.")

        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        conv_output = conv_output[0]
        heatmap = tf.reduce_sum(conv_output * pooled_gradients, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        maximum = tf.reduce_max(heatmap)
        heatmap = tf.where(maximum > 0, heatmap / maximum, heatmap)
        return heatmap.numpy()

    @staticmethod
    def _create_overlay_data_url(original: Image.Image, heatmap: np.ndarray) -> str:
        resized_heatmap = Image.fromarray(
            np.uint8(np.clip(heatmap, 0, 1) * 255), mode="L"
        ).resize(original.size, Image.Resampling.BILINEAR)
        heat = np.asarray(resized_heatmap, dtype=np.float32) / 255.0

        # Lightweight blue -> cyan -> yellow -> red medical heatmap.
        red = np.clip(2.2 * heat, 0, 1)
        green = np.clip(2.2 * heat - 0.45, 0, 1)
        blue = np.clip(1.35 - 2.0 * heat, 0, 1)
        colour_map = np.stack([red, green, blue], axis=-1) * 255.0

        original_array = np.asarray(original, dtype=np.float32)
        alpha = (0.58 * heat)[..., np.newaxis]
        overlay = original_array * (1 - alpha) + colour_map * alpha
        overlay_image = Image.fromarray(np.uint8(np.clip(overlay, 0, 255)))
        return ModelService._image_to_data_url(overlay_image)

    @staticmethod
    def _image_to_data_url(image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _display_name(class_name: str) -> str:
        return DISPLAY_NAMES.get(class_name.lower(), class_name.replace("_", " ").title())
