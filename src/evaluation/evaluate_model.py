"""Run the complete, repeatable test-set evaluation workflow."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from src.utils.runtime_environment import configure_runtime_environment

configure_runtime_environment()

import numpy as np
import tensorflow as tf

from src.evaluation.classification_report import write_classification_report
from src.evaluation.confusion_matrix import save_confusion_matrix
from src.evaluation.metrics import calculate_metrics
from src.evaluation.report_generator import (generate_pdf_report, save_per_class_accuracy, save_precision_recall_curve,
                                             save_prediction_distribution, save_roc_curve)
from src.preprocessing.data_pipeline import DataPipelineConfig, build_dataset
from src.preprocessing.inspect_dataset import load_expected_classes
from src.training.build_model import LABELS_FILE
from src.utils.logging_config import configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "biomedical_waste_mobilenetv2.keras"
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "evaluation"
LOG_FILE = PROJECT_ROOT / "logs" / "model_evaluation.log"


def evaluate(model_path: Path, batch_size: int = 32) -> dict[str, Path]:
    """Evaluate a saved Keras model and generate all Phase 8 artifacts."""
    logger = configure_logging(LOG_FILE).getChild("evaluation")
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model was not found: {model_path}")
    class_names = load_expected_classes(LABELS_FILE)
    dataset = build_dataset("test", DataPipelineConfig(batch_size=batch_size, cache_mode="none"), training=False)
    model = tf.keras.models.load_model(model_path)
    logger.info("Loaded model %s; evaluating %s test classes.", model_path, len(class_names))
    probabilities = model.predict(dataset, verbose=1)
    labels = np.concatenate([batch_labels.numpy() for _, batch_labels in dataset])
    loss = float(model.evaluate(dataset, verbose=0)[0])
    metrics = calculate_metrics(labels, probabilities, class_names, loss)
    metrics.update({"generated_at_utc": datetime.now(timezone.utc).isoformat(), "model_path": str(model_path.relative_to(PROJECT_ROOT))})
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": RESULTS_DIRECTORY / "metrics.json", "classification_report": RESULTS_DIRECTORY / "classification_report.txt",
        "confusion_matrix": RESULTS_DIRECTORY / "confusion_matrix.png", "per_class_accuracy": RESULTS_DIRECTORY / "per_class_accuracy.png",
        "roc_curve": RESULTS_DIRECTORY / "roc_curve.png", "precision_recall_curve": RESULTS_DIRECTORY / "precision_recall_curve.png",
        "prediction_distribution": RESULTS_DIRECTORY / "prediction_distribution.png", "pdf": RESULTS_DIRECTORY / "evaluation_report.pdf",
    }
    paths["metrics"].write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_classification_report(metrics, paths["classification_report"])
    save_confusion_matrix(metrics["confusion_matrix"], class_names, paths["confusion_matrix"])
    save_per_class_accuracy(metrics, paths["per_class_accuracy"])
    save_roc_curve(metrics, paths["roc_curve"])
    save_precision_recall_curve(metrics, paths["precision_recall_curve"])
    save_prediction_distribution(metrics, paths["prediction_distribution"])
    generate_pdf_report(metrics, paths, paths["pdf"])
    logger.info("Evaluation complete. Top-1 accuracy: %.4f", metrics["top_1_accuracy"])
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained biomedical-waste classifier on the test set.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    arguments = parser.parse_args()
    try:
        generated_paths = evaluate(arguments.model, arguments.batch_size)
    except (FileNotFoundError, OSError, ValueError, tf.errors.OpError) as error:
        configure_logging(LOG_FILE).getChild("evaluation").exception("Evaluation failed: %s", error)
        raise SystemExit(f"Evaluation failed: {error}") from error
    print(f"Evaluation report written to: {generated_paths['pdf']}")
