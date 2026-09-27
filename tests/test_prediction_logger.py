"""Tests for prediction audit CSV schema."""

import csv
import tempfile
import unittest
from pathlib import Path

from src.inference.predict_image import Prediction
from src.utils.prediction_logger import PredictionLogger


class PredictionLoggerTests(unittest.TestCase):
    def test_event_is_written_with_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv"
            PredictionLogger(path).log(Prediction("red", 1, 0.8, 12.5, (("red", 0.8),)), 1, "classified")
            with path.open(encoding="utf-8") as file:
                row = next(csv.DictReader(file))
            self.assertEqual(row["Prediction"], "Red")
            self.assertEqual(row["Command Sent"], "1")
            self.assertEqual(row["Status"], "classified")
