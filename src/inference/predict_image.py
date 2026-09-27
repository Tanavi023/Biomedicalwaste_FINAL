"""Predict one biomedical-waste image with the trained MobileNetV2 model.

This module contains the reusable predictor used by the folder and webcam
workflows.  It intentionally has no serial or hardware dependencies.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from src.utils.runtime_environment import configure_runtime_environment

configure_runtime_environment()

import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src.preprocessing.image_preprocessing import IMAGE_SIZE, preprocess_for_inference
from src.preprocessing.inspect_dataset import load_expected_classes
from src.training.build_model import LABELS_FILE
from src.utils.logging_config import configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "biomedical_waste_mobilenetv2.keras"
LOG_FILE = PROJECT_ROOT / "logs" / "inference.log"
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class Prediction:
    """A JSON-ready classification result for one image."""

    predicted_class: str
    class_index: int
    confidence: float
    inference_time_ms: float
    top_predictions: tuple[tuple[str, float], ...]

    def as_dict(self) -> dict[str, object]:
        """Return a stable, presentation-friendly result payload."""
        return {
            "predicted_class": self.predicted_class.title(),
            "class_index": self.class_index,
            "confidence": round(self.confidence, 6),
            "inference_time_ms": round(self.inference_time_ms, 3),
            "top_3_predictions": [
                {"class": class_name.title(), "confidence": round(confidence, 6)}
                for class_name, confidence in self.top_predictions
            ],
        }


class BiomedicalWastePredictor:
    """Load the model once and classify RGB images safely and consistently."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH, labels_path: Path = LABELS_FILE) -> None:
        self.logger = configure_logging(LOG_FILE).getChild("inference")
        if not model_path.is_file():
            raise FileNotFoundError(f"Trained model was not found: {model_path}")
        if not labels_path.is_file():
            raise FileNotFoundError(f"Labels file was not found: {labels_path}")
        self.class_names = load_expected_classes(labels_path)
        self.model = tf.keras.models.load_model(model_path, compile=False)
        output_classes = int(self.model.output_shape[-1])
        if output_classes != len(self.class_names):
            raise ValueError(
                f"Model outputs {output_classes} classes but labels define {len(self.class_names)} classes."
            )
        self.logger.info("Loaded inference model from %s", model_path)

    @staticmethod
    def _validate_rgb_image(image: np.ndarray) -> np.ndarray:
        """Validate and normalize an in-memory RGB image to uint8-compatible data."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image array with shape (height, width, 3).")
        if image.size == 0:
            raise ValueError("The image is empty.")
        if not np.isfinite(image).all():
            raise ValueError("The image contains non-finite pixel values.")
        return np.clip(image, 0, 255).astype(np.uint8, copy=False)

    @staticmethod
    def load_image(image_path: Path) -> np.ndarray:
        """Decode one supported image as RGB, rejecting corrupt or invalid files."""
        if not image_path.is_file():
            raise FileNotFoundError(f"Image was not found: {image_path}")
        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image type '{image_path.suffix}'. Supported types: {sorted(SUPPORTED_EXTENSIONS)}")
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                return np.asarray(image.convert("RGB"))
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise ValueError(f"Invalid or unreadable image: {image_path}") from error

    def predict_array(self, rgb_image: np.ndarray, top_k: int = 3) -> Prediction:
        """Classify an RGB NumPy image and measure model-only inference time."""
        image = self._validate_rgb_image(rgb_image)
        if not 1 <= top_k <= len(self.class_names):
            raise ValueError(f"top_k must be between 1 and {len(self.class_names)}.")
        tensor = preprocess_for_inference(tf.convert_to_tensor(image))[tf.newaxis, ...]
        started = time.perf_counter()
        probabilities = np.asarray(self.model(tensor, training=False))[0]
        elapsed_ms = (time.perf_counter() - started) * 1_000
        if probabilities.shape != (len(self.class_names),):
            raise ValueError(f"Unexpected model output shape: {probabilities.shape}")
        ranking = np.argsort(probabilities)[::-1][:top_k]
        class_index = int(ranking[0])
        return Prediction(
            predicted_class=self.class_names[class_index],
            class_index=class_index,
            confidence=float(probabilities[class_index]),
            inference_time_ms=elapsed_ms,
            top_predictions=tuple((self.class_names[int(index)], float(probabilities[index])) for index in ranking),
        )

    def predict_file(self, image_path: Path, top_k: int = 3) -> Prediction:
        """Load and classify one image file."""
        return self.predict_array(self.load_image(image_path), top_k=top_k)


def main() -> None:
    """Run the single-image command-line interface."""
    parser = argparse.ArgumentParser(description="Predict the biomedical waste category for one image.")
    parser.add_argument("image", type=Path, help="Path to a PNG, JPG, JPEG, BMP, or WEBP image.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the trained Keras model.")
    args = parser.parse_args()
    try:
        result = BiomedicalWastePredictor(args.model).predict_file(args.image)
    except (FileNotFoundError, ValueError, OSError, tf.errors.OpError) as error:
        configure_logging(LOG_FILE).getChild("inference").exception("Single-image prediction failed: %s", error)
        raise SystemExit(f"Prediction failed: {error}") from error
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
