"""Durable CSV audit trail for successful predictions and processing failures."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from src.inference.predict_image import Prediction

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "prediction_logs.csv"
FIELDNAMES = ["Timestamp", "Prediction", "Confidence", "Inference Time (ms)", "Command Sent", "Status"]


class PredictionLogger:
    """Append structured prediction and failure events to the existing CSV schema."""

    def __init__(self, path: Path = DEFAULT_LOG_PATH) -> None:
        self.path = path

    def _append(self, row: dict[str, str | int]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def log(self, prediction: Prediction, command: int | None, status: str) -> None:
        """Write one genuine model prediction."""
        self._append({
            "Timestamp": datetime.now(timezone.utc).isoformat(),
            "Prediction": prediction.predicted_class.title(),
            "Confidence": f"{prediction.confidence:.6f}",
            "Inference Time (ms)": f"{prediction.inference_time_ms:.3f}",
            "Command Sent": "" if command is None else command,
            "Status": status,
        })

    def log_failure(self, object_id: int | str | None, message: str) -> None:
        """Write a failure without inventing a model class or confidence."""
        safe_id = "unknown" if object_id is None else str(object_id).replace("\r", " ").replace("\n", " ")[:128]
        detail = " ".join(str(message).replace("\r", " ").replace("\n", " ").split())[:180]
        self._append({
            "Timestamp": datetime.now(timezone.utc).isoformat(),
            "Prediction": "Unknown",
            "Confidence": "",
            "Inference Time (ms)": "",
            "Command Sent": "",
            "Status": f"failed; object_id={safe_id}; {detail}"[:240],
        })

    def clear(self) -> None:
        """Reset the audit file to its header after user confirmation in the UI."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=FIELDNAMES).writeheader()
