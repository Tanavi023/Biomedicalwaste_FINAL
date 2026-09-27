"""Durable CSV audit trail for model decisions and hardware outcomes."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from src.inference.predict_image import Prediction

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = PROJECT_ROOT / "logs" / "prediction_logs.csv"
FIELDNAMES = ["Timestamp", "Prediction", "Confidence", "Inference Time (ms)", "Command Sent", "Status"]


class PredictionLogger:
    """Append structured prediction events without loading the full log file."""

    def __init__(self, path: Path = DEFAULT_LOG_PATH) -> None:
        self.path = path

    def log(self, prediction: Prediction, command: int | None, status: str) -> None:
        """Write one auditable event; ``command`` stays blank when disconnected."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow({
                "Timestamp": datetime.now(timezone.utc).isoformat(), "Prediction": prediction.predicted_class.title(),
                "Confidence": f"{prediction.confidence:.6f}", "Inference Time (ms)": f"{prediction.inference_time_ms:.3f}",
                "Command Sent": "" if command is None else command, "Status": status,
            })

    def clear(self) -> None:
        """Reset the audit file to its header after user confirmation in the UI."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as file:
            csv.DictWriter(file, fieldnames=FIELDNAMES).writeheader()
