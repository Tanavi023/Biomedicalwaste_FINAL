"""Create a visual audit of preprocessing and training augmentation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.utils.runtime_environment import configure_runtime_environment

configure_runtime_environment()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

from src.preprocessing.image_preprocessing import (
    RANDOM_SEED,
    build_augmentation_layer,
    denormalize_for_display,
    preprocess_for_inference,
    resize_image,
)
from src.preprocessing.inspect_dataset import load_expected_classes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DATA_DIR = PROJECT_ROOT / "data" / "train"
LABELS_FILE = PROJECT_ROOT / "data" / "labels.json"
OUTPUT_PATH = PROJECT_ROOT / "results" / "preprocessing" / "preprocessing_preview.png"
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def load_rgb_image(path: Path) -> tf.Tensor:
    """Read one image as a variable-size RGB tensor with pixel range 0-255."""
    encoded = tf.io.read_file(str(path))
    image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return tf.cast(image, tf.float32)


def select_samples(samples_per_class: int, seed: int) -> list[tuple[str, Path]]:
    """Select reproducible, class-balanced examples from the training split."""
    random_generator = random.Random(seed)
    selected_samples: list[tuple[str, Path]] = []
    for class_name in load_expected_classes(LABELS_FILE):
        candidates = sorted(
            path
            for path in (TRAIN_DATA_DIR / class_name).iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if len(candidates) < samples_per_class:
            raise ValueError(f"Class '{class_name}' has fewer than {samples_per_class} training images.")
        selected_samples.extend((class_name, path) for path in random_generator.sample(candidates, samples_per_class))
    return selected_samples


def create_preview(samples_per_class: int, seed: int, output_path: Path) -> Path:
    """Save a grid of resized originals and two independently augmented views."""
    samples = select_samples(samples_per_class, seed)
    augmentation_layer = build_augmentation_layer(seed)
    figure, axes = plt.subplots(len(samples), 3, figsize=(12, 3.2 * len(samples)), squeeze=False)

    for row_index, (class_name, image_path) in enumerate(samples):
        image = load_rgb_image(image_path)
        resized = resize_image(image)
        augmented_first = augmentation_layer(resized, training=True)
        augmented_second = augmentation_layer(resized, training=True)
        display_images = [
            tf.clip_by_value(resized / 255.0, 0.0, 1.0),
            denormalize_for_display(preprocess_for_inference(augmented_first)),
            denormalize_for_display(preprocess_for_inference(augmented_second)),
        ]
        titles = ["Resized original", "Augmented view A", "Augmented view B"]
        for column_index, (display_image, title) in enumerate(zip(display_images, titles, strict=True)):
            axes[row_index, column_index].imshow(display_image.numpy())
            axes[row_index, column_index].set_title(f"{title}\n{class_name.title()}")
            axes[row_index, column_index].axis("off")

    figure.suptitle("MobileNetV2 preprocessing and training-only augmentation", fontsize=16)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse visual-audit settings."""
    parser = argparse.ArgumentParser(description="Save a preprocessing and augmentation preview image.")
    parser.add_argument("--samples-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    saved_path = create_preview(args.samples_per_class, args.seed, args.output)
    print(f"Preprocessing preview saved to: {saved_path}")
