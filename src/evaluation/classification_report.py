"""Human-readable classification-report formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_classification_report(metrics: dict[str, Any], output_path: Path) -> Path:
    """Write a compact text report with class and aggregate metrics."""
    lines = ["Biomedical Waste Classifier - Test Evaluation", "=" * 58, "", "Per-class metrics", "-" * 58]
    lines.append(f"{'Class':<12}{'Precision':>11}{'Recall':>11}{'F1':>11}{'Specificity':>13}{'Support':>10}")
    for class_name, values in metrics["per_class"].items():
        lines.append(
            f"{class_name.title():<12}{values['precision']:>11.4f}{values['recall']:>11.4f}"
            f"{values['f1_score']:>11.4f}{values['specificity']:>13.4f}{values['support']:>10}"
        )
    lines.extend(["", "Overall metrics", "-" * 58])
    lines.extend(f"{label:<24}{value:.4f}" for label, value in (
        ("Test loss", metrics["loss"]), ("Test accuracy / Top-1", metrics["top_1_accuracy"]),
        ("Top-2 accuracy", metrics["top_2_accuracy"]), ("Macro precision", metrics["macro_average"]["precision"]),
        ("Macro recall", metrics["macro_average"]["recall"]), ("Macro F1", metrics["macro_average"]["f1_score"]),
        ("Macro specificity", metrics["macro_average"]["specificity"]),
    ))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path
