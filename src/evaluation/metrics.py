"""Dependency-light metric calculations for multiclass classification."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy import trapezoid


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Return a rows=true, columns=predicted confusion matrix."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true.astype(int), y_pred.astype(int)), 1)
    return matrix


def _safe_divide(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray | float:
    """Divide safely, returning zero when a metric is undefined."""
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=np.asarray(denominator) != 0)


def per_class_metrics(matrix: np.ndarray, class_names: list[str]) -> dict[str, dict[str, float | int]]:
    """Calculate one-vs-rest precision, recall, F1, specificity, and accuracy."""
    total = int(matrix.sum())
    values: dict[str, dict[str, float | int]] = {}
    for index, class_name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        true_negative = total - true_positive - false_positive - false_negative
        precision = float(_safe_divide(true_positive, true_positive + false_positive))
        recall = float(_safe_divide(true_positive, true_positive + false_negative))
        values[class_name] = {
            "support": int(matrix[index, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1_score": float(_safe_divide(2 * precision * recall, precision + recall)),
            "specificity": float(_safe_divide(true_negative, true_negative + false_positive)),
            "accuracy": float(_safe_divide(true_positive + true_negative, total)),
        }
    return values


def _binary_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an ROC-compatible threshold curve without scikit-learn."""
    order = np.argsort(scores, kind="mergesort")[::-1]
    labels, ordered_scores = y_true[order].astype(int), scores[order]
    distinct = np.where(np.diff(ordered_scores))[0]
    indices = np.r_[distinct, len(labels) - 1]
    true_positives = np.cumsum(labels)[indices]
    false_positives = 1 + indices - true_positives
    positives, negatives = int(labels.sum()), int(len(labels) - labels.sum())
    tpr = np.r_[0.0, _safe_divide(true_positives, positives), 1.0]
    fpr = np.r_[0.0, _safe_divide(false_positives, negatives), 1.0]
    thresholds = np.r_[np.inf, ordered_scores[indices], -np.inf]
    return fpr.astype(float), tpr.astype(float), thresholds.astype(float)


def roc_pr_metrics(y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    """Return per-class one-vs-rest ROC and precision-recall data and AUCs."""
    curves: dict[str, Any] = {}
    for index, class_name in enumerate(class_names):
        binary_truth = (y_true == index).astype(int)
        fpr, tpr, _ = _binary_curve(binary_truth, probabilities[:, index])
        # Precision-recall is evaluated at every descending score threshold.
        order = np.argsort(probabilities[:, index], kind="mergesort")[::-1]
        ranked_truth = binary_truth[order]
        true_positives = np.cumsum(ranked_truth)
        precision = _safe_divide(true_positives, np.arange(1, len(ranked_truth) + 1)).astype(float)
        recall = _safe_divide(true_positives, max(int(binary_truth.sum()), 1)).astype(float)
        precision = np.r_[1.0, precision]
        recall = np.r_[0.0, recall]
        curves[class_name] = {
            "roc_auc": float(trapezoid(tpr, fpr)),
            "pr_auc": float(np.trapezoid(precision, recall)),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "precision_curve": precision.tolist(),
            "recall_curve": recall.tolist(),
        }
    return curves


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray, class_names: list[str], loss: float) -> dict[str, Any]:
    """Create a JSON-ready, comprehensive evaluation metrics dictionary."""
    y_pred = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y_true, y_pred, len(class_names))
    per_class = per_class_metrics(matrix, class_names)
    supports = np.array([per_class[name]["support"] for name in class_names], dtype=float)
    metric_names = ("precision", "recall", "f1_score", "specificity")
    macro = {name: float(np.mean([per_class[item][name] for item in class_names])) for name in metric_names}
    weighted = {
        name: float(np.average([per_class[item][name] for item in class_names], weights=supports)) for name in metric_names
    }
    top_1 = float(np.mean(y_pred == y_true))
    top_2 = float(np.mean(np.any(np.argsort(probabilities, axis=1)[:, -2:] == y_true[:, None], axis=1)))
    return {
        "test_samples": int(len(y_true)),
        "loss": float(loss),
        "test_accuracy": top_1,
        "top_1_accuracy": top_1,
        "top_2_accuracy": top_2,
        "macro_average": macro,
        "weighted_average": weighted,
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "prediction_distribution": {class_name: int(np.sum(y_pred == index)) for index, class_name in enumerate(class_names)},
        "true_distribution": {class_name: int(np.sum(y_true == index)) for index, class_name in enumerate(class_names)},
        "roc_pr": roc_pr_metrics(y_true, probabilities, class_names),
    }
