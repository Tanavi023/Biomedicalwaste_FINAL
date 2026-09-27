"""Build the MobileNetV2 transfer-learning classifier architecture."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.utils.runtime_environment import configure_runtime_environment

configure_runtime_environment()

import tensorflow as tf

from src.preprocessing.image_preprocessing import IMAGE_SHAPE
from src.preprocessing.inspect_dataset import load_expected_classes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS_FILE = PROJECT_ROOT / "data" / "labels.json"
SUMMARY_PATH = PROJECT_ROOT / "results" / "model_architecture" / "mobilenetv2_transfer_learning_summary.txt"


@dataclass(frozen=True)
class ModelConfig:
    """Architecture settings for the four-class transfer-learning model."""

    input_shape: tuple[int, int, int] = IMAGE_SHAPE
    dense_units: int = 128
    dropout_rate: float = 0.30
    l2_regularization: float = 1e-4


def build_classifier(
    num_classes: int,
    config: ModelConfig = ModelConfig(),
    weights: str | None = "imagenet",
) -> tf.keras.Model:
    """Create an uncompiled MobileNetV2 classifier with a frozen backbone.

    Inputs must already be RGB float32 tensors normalized to MobileNetV2's
    ``[-1, 1]`` range. Compilation intentionally belongs to the training phase.
    """
    if num_classes < 2:
        raise ValueError("A classifier requires at least two output classes.")
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=config.input_shape,
        include_top=False,
        weights=weights,
        name="mobilenetv2_backbone",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=config.input_shape, name="image")
    features = backbone(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(features)
    features = tf.keras.layers.Dropout(config.dropout_rate, name="head_dropout_1")(features)
    features = tf.keras.layers.Dense(
        config.dense_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(config.l2_regularization),
        name="classification_dense",
    )(features)
    features = tf.keras.layers.Dropout(config.dropout_rate / 2, name="head_dropout_2")(features)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_probabilities")(features)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="biomedical_waste_mobilenetv2")


def build_project_classifier(weights: str | None = "imagenet") -> tf.keras.Model:
    """Build the classifier using the class count defined in labels.json."""
    return build_classifier(num_classes=len(load_expected_classes(LABELS_FILE)), weights=weights)


def write_model_summary(model: tf.keras.Model, output_path: Path = SUMMARY_PATH) -> Path:
    """Write the Keras architecture summary to a versionable text artifact."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    model.summary(print_fn=lines.append, expand_nested=True, show_trainable=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse architecture-summary options."""
    parser = argparse.ArgumentParser(description="Build and summarize the MobileNetV2 transfer-learning classifier.")
    parser.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    parser.add_argument("--summary-path", type=Path, default=SUMMARY_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selected_weights = None if args.weights == "none" else args.weights
    model = build_project_classifier(weights=selected_weights)
    summary_path = write_model_summary(model, args.summary_path)
    print(f"Model summary written to: {summary_path}")
