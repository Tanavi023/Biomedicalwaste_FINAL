"""Build efficient, reproducible TensorFlow datasets from split image folders."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Mapping

from src.utils.runtime_environment import configure_runtime_environment

configure_runtime_environment()

import tensorflow as tf

from src.preprocessing.image_preprocessing import (
    IMAGE_SHAPE,
    IMAGE_SIZE,
    RANDOM_SEED,
    build_augmentation_layer,
    normalize_for_mobilenet_v2,
)
from src.preprocessing.inspect_dataset import load_expected_classes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS_FILE = PROJECT_ROOT / "data" / "labels.json"
SPLIT_DIRECTORIES = {
    "train": PROJECT_ROOT / "data" / "train",
    "validation": PROJECT_ROOT / "data" / "validation",
    "test": PROJECT_ROOT / "data" / "test",
}
CACHE_DIRECTORY = PROJECT_ROOT / "data" / "cache" / "tfdata"
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "data_pipeline"
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png"}
CacheMode = Literal["none", "memory", "disk"]


@dataclass(frozen=True)
class DataPipelineConfig:
    """Settings governing image loading, batching, cache behavior, and shuffling."""

    batch_size: int = 32
    seed: int = RANDOM_SEED
    cache_mode: CacheMode = "disk"
    shuffle_buffer_size: int = 1_000


def count_images(directory: Path) -> int:
    """Count image files in one directory tree using Keras-supported formats."""
    return sum(1 for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)


def apply_cache(dataset: tf.data.Dataset, split_name: str, cache_mode: CacheMode) -> tf.data.Dataset:
    """Cache decoded, resized images before random augmentation or normalization."""
    if cache_mode == "none":
        return dataset
    if cache_mode == "memory":
        return dataset.cache()
    if cache_mode == "disk":
        CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        return dataset.cache(str(CACHE_DIRECTORY / split_name))
    raise ValueError(f"Unsupported cache mode: {cache_mode}")


def build_dataset(split_name: str, config: DataPipelineConfig, training: bool) -> tf.data.Dataset:
    """Return one batched, normalized, prefetched dataset for a named split.

    Images are decoded and resized by ``image_dataset_from_directory``. Caching
    happens before random augmentation, which allows different augmented views
    every epoch without repeatedly decoding source files.
    """
    if split_name not in SPLIT_DIRECTORIES:
        raise ValueError(f"Unknown split '{split_name}'. Expected one of {sorted(SPLIT_DIRECTORIES)}.")
    directory = SPLIT_DIRECTORIES[split_name]
    class_names = load_expected_classes(LABELS_FILE)
    image_count = count_images(directory)
    if image_count == 0:
        raise ValueError(f"No supported images found in {directory}.")

    dataset = tf.keras.utils.image_dataset_from_directory(
        directory=directory,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        color_mode="rgb",
        batch_size=config.batch_size,
        image_size=IMAGE_SIZE,
        shuffle=False,
        interpolation="bilinear",
        verbose=False,
    )
    dataset = dataset.map(
        lambda images, labels: (tf.cast(images, tf.float32), tf.cast(labels, tf.int32)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = apply_cache(dataset, split_name, config.cache_mode)

    if training:
        augmentation_layer = build_augmentation_layer(config.seed)
        dataset = dataset.shuffle(
            buffer_size=min(config.shuffle_buffer_size, image_count),
            seed=config.seed,
            reshuffle_each_iteration=True,
        )
        dataset = dataset.map(
            lambda images, labels: (normalize_for_mobilenet_v2(augmentation_layer(images, training=True)), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    else:
        dataset = dataset.map(
            lambda images, labels: (normalize_for_mobilenet_v2(images), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_all_datasets(
    config: DataPipelineConfig = DataPipelineConfig(),
    cache_modes: Mapping[str, CacheMode] | None = None,
) -> dict[str, tf.data.Dataset]:
    """Build the training, validation, and test datasets with explicit cache modes."""
    selected_cache_modes = cache_modes or {name: config.cache_mode for name in SPLIT_DIRECTORIES}
    if set(selected_cache_modes) != set(SPLIT_DIRECTORIES):
        raise ValueError("cache_modes must define train, validation, and test values.")
    return {
        "train": build_dataset("train", DataPipelineConfig(**{**config.__dict__, "cache_mode": selected_cache_modes["train"]}), training=True),
        "validation": build_dataset("validation", DataPipelineConfig(**{**config.__dict__, "cache_mode": selected_cache_modes["validation"]}), training=False),
        "test": build_dataset("test", DataPipelineConfig(**{**config.__dict__, "cache_mode": selected_cache_modes["test"]}), training=False),
    }


def run_smoke_test(batch_size: int = 32) -> Path:
    """Validate one batch from each split without creating persistent cache files."""
    config = DataPipelineConfig(batch_size=batch_size, cache_mode="none")
    datasets = build_all_datasets(config, cache_modes={name: "none" for name in SPLIT_DIRECTORIES})
    report: dict[str, object] = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_size": batch_size,
        "image_shape": IMAGE_SHAPE,
        "splits": {},
    }
    for split_name, dataset in datasets.items():
        images, labels = next(iter(dataset))
        tf.debugging.assert_rank(images, 4)
        tf.debugging.assert_equal(tf.shape(images)[1:], IMAGE_SHAPE)
        tf.debugging.assert_all_finite(images, "Image batch contains non-finite values.")
        tf.debugging.assert_greater_equal(images, -1.0)
        tf.debugging.assert_less_equal(images, 1.0)
        report["splits"][split_name] = {
            "batch_shape": list(images.shape),
            "label_shape": list(labels.shape),
            "label_dtype": labels.dtype.name,
            "pixel_min": float(tf.reduce_min(images)),
            "pixel_max": float(tf.reduce_max(images)),
        }
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIRECTORY / "pipeline_smoke_test.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    """Parse the optional smoke-test batch size."""
    parser = argparse.ArgumentParser(description="Build and verify TensorFlow image datasets.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.smoke_test:
        print(f"Pipeline smoke test saved to: {run_smoke_test(args.batch_size)}")
