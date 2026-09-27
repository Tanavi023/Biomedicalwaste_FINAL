"""Benchmark the Keras model and export a TensorFlow Lite deployment artifact."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import psutil

from src.inference.predict_image import DEFAULT_MODEL_PATH, BiomedicalWastePredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "performance"
DEFAULT_TFLITE_PATH = PROJECT_ROOT / "models" / "biomedical_waste_mobilenetv2.tflite"


def benchmark(model_path: Path, sample_image: Path, runs: int = 20) -> Path:
    """Measure steady-state inference latency, CPU delta, memory, and model size."""
    if runs < 1:
        raise ValueError("runs must be at least 1.")
    predictor = BiomedicalWastePredictor(model_path)
    image = predictor.load_image(sample_image)
    predictor.predict_array(image)  # TensorFlow warm-up; excluded from measurement.
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss
    cpu_before = process.cpu_times()
    started = time.perf_counter()
    timings = [predictor.predict_array(image).inference_time_ms for _ in range(runs)]
    wall_seconds = time.perf_counter() - started
    cpu_after = process.cpu_times()
    report = {
        "model_path": str(model_path.relative_to(PROJECT_ROOT)), "sample_image": str(sample_image), "runs": runs,
        "mean_inference_time_ms": round(sum(timings) / len(timings), 3), "min_inference_time_ms": round(min(timings), 3),
        "max_inference_time_ms": round(max(timings), 3), "estimated_fps": round(runs / wall_seconds, 2),
        "process_memory_mb": round(process.memory_info().rss / 1024 / 1024, 2),
        "memory_growth_mb": round((process.memory_info().rss - memory_before) / 1024 / 1024, 2),
        "cpu_seconds": round((cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system), 3),
        "keras_model_size_mb": round(model_path.stat().st_size / 1024 / 1024, 2),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = RESULTS_DIR / "benchmark.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return destination


def export_tflite(model_path: Path, output_path: Path = DEFAULT_TFLITE_PATH) -> Path:
    """Export the saved Keras model with default size-optimizing conversion."""
    from src.utils.runtime_environment import configure_runtime_environment
    configure_runtime_environment()
    import tensorflow as tf

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model was not found: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(converter.convert())
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark or export the biomedical-waste model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--benchmark-image", type=Path)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--export-tflite", action="store_true")
    args = parser.parse_args()
    if args.benchmark_image:
        print(f"Benchmark written to: {benchmark(args.model, args.benchmark_image, args.runs)}")
    if args.export_tflite:
        print(f"TensorFlow Lite model written to: {export_tflite(args.model)}")
    if not args.benchmark_image and not args.export_tflite:
        parser.error("Specify --benchmark-image and/or --export-tflite.")


if __name__ == "__main__":
    main()
