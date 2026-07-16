from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Keras model for NeuroVision AI.")
    parser.add_argument("model", nargs="?", default="models/best_model.keras")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    print("Loaded:", model_path)
    print("Input shape:", model.input_shape)
    print("Output shape:", model.output_shape)
    print("Layers:", len(model.layers))
    print("Parameters:", f"{model.count_params():,}")
    if int(model.output_shape[-1]) != 4:
        raise SystemExit("Expected 4 output units.")
    print("Model is compatible with the four-class application.")


if __name__ == "__main__":
    main()
