"""Train and evaluate the biomedical-waste MobileNetV2 classifier."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.utils.runtime_environment import configure_runtime_environment

configure_runtime_environment()

import numpy as np
import tensorflow as tf

from src.preprocessing.data_pipeline import DataPipelineConfig, build_all_datasets, count_images
from src.preprocessing.image_preprocessing import RANDOM_SEED
from src.preprocessing.inspect_dataset import load_expected_classes
from src.training.build_model import LABELS_FILE, build_project_classifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "biomedical_waste_mobilenetv2.keras"
HEAD_CANDIDATE_PATH = PROJECT_ROOT / "models" / "biomedical_waste_mobilenetv2_head.keras"
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "training"


@dataclass(frozen=True)
class TrainingConfig:
    """Settings for reproducible transfer learning."""

    batch_size: int = 32
    head_epochs: int = 8
    fine_tune_epochs: int = 4
    head_learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    fine_tune_from_layer: int = 120
    seed: int = RANDOM_SEED


def set_random_seed(seed: int) -> None:
    """Set the random state used by Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def class_weights(class_names: list[str]) -> dict[int, float]:
    """Calculate balanced weights from the training split only."""
    counts = [count_images(PROJECT_ROOT / "data" / "train" / class_name) for class_name in class_names]
    total = sum(counts)
    if not total or 0 in counts:
        raise ValueError("Every class must contain at least one training image.")
    return {index: total / (len(counts) * count) for index, count in enumerate(counts)}


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    """Compile with sparse labels and evaluation metrics suitable for four classes."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy", tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top_2_accuracy")],
    )


def callbacks(model_path: Path) -> list[tf.keras.callbacks.Callback]:
    """Return validation-loss callbacks and persist the best complete model."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path), monitor="val_loss", mode="min", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=3, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", patience=1, factor=0.2, min_lr=1e-7, verbose=1),
    ]


def merge_history(*histories: tf.keras.callbacks.History) -> dict[str, list[float]]:
    """Combine sequential Keras history objects into JSON-ready values."""
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(value) for value in values)
    return merged


def train(config: TrainingConfig) -> Path:
    """Train frozen-head then fine-tuned models and write a metrics report."""
    set_random_seed(config.seed)
    class_names = load_expected_classes(LABELS_FILE)
    datasets = build_all_datasets(DataPipelineConfig(batch_size=config.batch_size, seed=config.seed))
    weights = class_weights(class_names)

    model = build_project_classifier(weights="imagenet")
    compile_model(model, config.head_learning_rate)
    head_history = model.fit(
        datasets["train"], validation_data=datasets["validation"], epochs=config.head_epochs,
        class_weight=weights, callbacks=callbacks(HEAD_CANDIDATE_PATH), verbose=2,
    )
    head_validation = model.evaluate(datasets["validation"], return_dict=True, verbose=0)

    backbone = model.get_layer("mobilenetv2_backbone")
    backbone.trainable = True
    for layer in backbone.layers[: config.fine_tune_from_layer]:
        layer.trainable = False
    compile_model(model, config.fine_tune_learning_rate)
    fine_tune_history = model.fit(
        datasets["train"], validation_data=datasets["validation"], initial_epoch=len(head_history.epoch),
        epochs=len(head_history.epoch) + config.fine_tune_epochs, class_weight=weights,
        callbacks=callbacks(MODEL_PATH), verbose=2,
    )
    fine_tune_validation = model.evaluate(datasets["validation"], return_dict=True, verbose=0)
    if head_validation["loss"] <= fine_tune_validation["loss"]:
        best_model = tf.keras.models.load_model(HEAD_CANDIDATE_PATH)
        best_phase = "head_training"
    else:
        best_model = tf.keras.models.load_model(MODEL_PATH)
        best_phase = "fine_tuning"
    best_model.save(MODEL_PATH)
    HEAD_CANDIDATE_PATH.unlink(missing_ok=True)
    test_metrics = best_model.evaluate(datasets["test"], return_dict=True, verbose=0)
    report = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "classes": class_names,
        "class_weights": weights,
        "history": merge_history(head_history, fine_tune_history),
        "validation_metrics": {
            "head_training": {name: float(value) for name, value in head_validation.items()},
            "fine_tuning": {name: float(value) for name, value in fine_tune_validation.items()},
        },
        "selected_phase": best_phase,
        "test_metrics": {name: float(value) for name, value in test_metrics.items()},
        "best_model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
    }
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIRECTORY / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the biomedical-waste transfer-learning classifier.")
    parser.add_argument("--head-epochs", type=int, default=TrainingConfig.head_epochs)
    parser.add_argument("--fine-tune-epochs", type=int, default=TrainingConfig.fine_tune_epochs)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    report_path = train(TrainingConfig(batch_size=args.batch_size, head_epochs=args.head_epochs, fine_tune_epochs=args.fine_tune_epochs))
    print(f"Training report written to: {report_path}")
