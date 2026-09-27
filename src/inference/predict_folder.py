"""Batch prediction workflow that exports one row per input image."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.inference.predict_image import (
    DEFAULT_MODEL_PATH,
    LOG_FILE,
    SUPPORTED_EXTENSIONS,
    BiomedicalWastePredictor,
)
from src.utils.logging_config import configure_logging


def predict_folder(folder_path: Path, model_path: Path = DEFAULT_MODEL_PATH, recursive: bool = False) -> Path:
    """Predict supported images in a folder and save results, including failures."""
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Image folder was not found: {folder_path}")
    predictor = BiomedicalWastePredictor(model_path)
    paths = folder_path.rglob("*") if recursive else folder_path.glob("*")
    image_paths = sorted(path for path in paths if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not image_paths:
        raise ValueError(f"No supported images found in: {folder_path}")
    output_path = folder_path / "prediction_results.csv"
    fields = ["Filename", "Prediction", "Confidence", "Inference Time (ms)", "Status", "Error"]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for image_path in image_paths:
            try:
                result = predictor.predict_file(image_path)
                writer.writerow({
                    "Filename": str(image_path.relative_to(folder_path)), "Prediction": result.predicted_class.title(),
                    "Confidence": f"{result.confidence:.6f}", "Inference Time (ms)": f"{result.inference_time_ms:.3f}",
                    "Status": "success", "Error": "",
                })
            except (FileNotFoundError, ValueError, OSError) as error:
                predictor.logger.warning("Skipping %s: %s", image_path, error)
                writer.writerow({"Filename": str(image_path.relative_to(folder_path)), "Prediction": "", "Confidence": "",
                                 "Inference Time (ms)": "", "Status": "failed", "Error": str(error)})
    predictor.logger.info("Batch prediction complete: %s images exported to %s", len(image_paths), output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict all supported images in a folder.")
    parser.add_argument("folder", type=Path)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--recursive", action="store_true", help="Include images in subfolders.")
    args = parser.parse_args()
    try:
        print(f"Batch results written to: {predict_folder(args.folder, args.model, args.recursive)}")
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as error:
        configure_logging(LOG_FILE).getChild("batch_inference").exception("Batch prediction failed: %s", error)
        raise SystemExit(f"Batch prediction failed: {error}") from error


if __name__ == "__main__":
    main()
