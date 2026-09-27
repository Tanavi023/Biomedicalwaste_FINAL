"""Charts and professional PDF generation for model evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def save_per_class_accuracy(metrics: dict[str, Any], output_path: Path) -> Path:
    """Save recall-based per-class accuracy; this equals class-wise accuracy."""
    names = list(metrics["per_class"])
    scores = [metrics["per_class"][name]["recall"] for name in names]
    figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    bars = axis.bar([name.title() for name in names], scores, color="#1976D2")
    axis.set(title="Per-class accuracy (recall)", ylabel="Accuracy", ylim=(0, 1.05))
    axis.bar_label(bars, labels=[f"{score:.1%}" for score in scores], padding=3)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_roc_curve(metrics: dict[str, Any], output_path: Path) -> Path:
    """Save one-vs-rest ROC curves for all waste classes."""
    figure, axis = plt.subplots(figsize=(7, 5.5), layout="constrained")
    for name, values in metrics["roc_pr"].items():
        axis.plot(values["fpr"], values["tpr"], label=f"{name.title()} (AUC={values['roc_auc']:.3f})")
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    axis.set(title="One-vs-rest ROC curves", xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.05))
    axis.legend(loc="lower right")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_precision_recall_curve(metrics: dict[str, Any], output_path: Path) -> Path:
    """Save one-vs-rest precision-recall curves for all waste classes."""
    figure, axis = plt.subplots(figsize=(7, 5.5), layout="constrained")
    for name, values in metrics["roc_pr"].items():
        axis.plot(values["recall_curve"], values["precision_curve"], label=f"{name.title()} (AUC={values['pr_auc']:.3f})")
    axis.set(title="One-vs-rest precision-recall curves", xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.05))
    axis.legend(loc="lower left")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_prediction_distribution(metrics: dict[str, Any], output_path: Path) -> Path:
    """Compare ground-truth and predicted class counts."""
    names = list(metrics["true_distribution"])
    positions = np.arange(len(names))
    figure, axis = plt.subplots(figsize=(8, 4.8), layout="constrained")
    axis.bar(positions - 0.19, [metrics["true_distribution"][name] for name in names], 0.38, label="True", color="#455A64")
    axis.bar(positions + 0.19, [metrics["prediction_distribution"][name] for name in names], 0.38, label="Predicted", color="#26A69A")
    axis.set(title="Prediction distribution", ylabel="Image count", xticks=positions, xticklabels=[name.title() for name in names])
    axis.legend()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path


def generate_pdf_report(metrics: dict[str, Any], image_paths: dict[str, Path], output_path: Path) -> Path:
    """Create a polished, self-contained PDF summary of test-set results."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Subtitle", parent=styles["BodyText"], textColor=colors.HexColor("#455A64"), spaceAfter=12))
    document = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
    story: list[Any] = [Paragraph("Biomedical Waste Classifier", styles["Title"]), Paragraph("Test-set model evaluation report", styles["Subtitle"])]
    summary = [
        ["Metric", "Result"], ["Test samples", str(metrics["test_samples"])], ["Test loss", f"{metrics['loss']:.4f}"],
        ["Top-1 accuracy", f"{metrics['top_1_accuracy']:.2%}"], ["Top-2 accuracy", f"{metrics['top_2_accuracy']:.2%}"],
        ["Macro precision", f"{metrics['macro_average']['precision']:.2%}"], ["Macro recall", f"{metrics['macro_average']['recall']:.2%}"],
        ["Macro F1 score", f"{metrics['macro_average']['f1_score']:.2%}"], ["Macro specificity", f"{metrics['macro_average']['specificity']:.2%}"],
    ]
    table = Table(summary, colWidths=[2.8 * inch, 2.2 * inch])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0D47A1")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B0BEC5")), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F9FC")]), ("PADDING", (0, 0), (-1, -1), 7)]))
    story.extend([table, Spacer(1, 18), Paragraph("Per-class metrics", styles["Heading2"])])
    rows = [["Class", "Precision", "Recall", "F1", "Specificity", "Support"]]
    for name, values in metrics["per_class"].items():
        rows.append([name.title(), *(f"{values[key]:.2%}" for key in ("precision", "recall", "f1_score", "specificity")), str(values["support"])])
    class_table = Table(rows, repeatRows=1, colWidths=[1.0 * inch, 0.85 * inch, 0.75 * inch, 0.75 * inch, 1.0 * inch, 0.65 * inch])
    class_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1565C0")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#CFD8DC")), ("ALIGN", (1, 0), (-1, -1), "RIGHT"), ("PADDING", (0, 0), (-1, -1), 5)]))
    story.extend([class_table, PageBreak()])
    for title, key in (("Confusion matrix", "confusion_matrix"), ("Per-class accuracy", "per_class_accuracy"), ("ROC curves", "roc_curve"), ("Precision-recall curves", "precision_recall_curve"), ("Prediction distribution", "prediction_distribution")):
        story.extend([Paragraph(title, styles["Heading2"]), Image(str(image_paths[key]), width=6.2 * inch, height=4.0 * inch), Spacer(1, 12)])
    document.build(story)
    return output_path
