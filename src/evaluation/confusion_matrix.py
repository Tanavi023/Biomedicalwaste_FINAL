"""Confusion-matrix visualization for the evaluation report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix(matrix: list[list[int]] | np.ndarray, class_names: list[str], output_path: Path) -> Path:
    """Save a labelled rows=true, columns=predicted confusion-matrix chart."""
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(8, 6.5), layout="constrained")
    image = axis.imshow(values, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, label="Image count")
    axis.set(xticks=range(len(class_names)), yticks=range(len(class_names)), xticklabels=class_names,
             yticklabels=class_names, xlabel="Predicted class", ylabel="True class", title="Test-set confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")
    threshold = values.max() / 2 if values.size else 0
    for row, column in np.ndindex(values.shape):
        axis.text(column, row, str(values[row, column]), ha="center", va="center",
                  color="white" if values[row, column] > threshold else "black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output_path
