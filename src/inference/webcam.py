"""Interactive OpenCV webcam inference for biomedical-waste segregation."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

from src.inference.predict_image import DEFAULT_MODEL_PATH, LOG_FILE, BiomedicalWastePredictor
from src.utils.logging_config import configure_logging


def run_webcam(camera_index: int = 0, model_path=DEFAULT_MODEL_PATH, history_size: int = 5) -> None:
    """Open a camera, overlay live predictions, and release every resource safely.

    Controls: Q/Esc exits; H toggles prediction history.
    """
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required. Install dependencies from requirements.txt.") from error
    predictor = BiomedicalWastePredictor(model_path)
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open webcam index {camera_index}.")
    history: deque[str] = deque(maxlen=history_size)
    show_history = True
    last_time = time.perf_counter()
    try:
        while True:
            success, frame = capture.read()
            if not success:
                raise RuntimeError("Unable to read a frame from the webcam.")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = predictor.predict_array(rgb_frame)
            history.appendleft(f"{result.predicted_class.title()}: {result.confidence:.1%}")
            now = time.perf_counter()
            fps = 1 / max(now - last_time, 1e-6)
            last_time = now
            height, width = frame.shape[:2]
            cv2.rectangle(frame, (10, 10), (width - 10, height - 10), (0, 200, 0), 2)
            cv2.putText(frame, f"{result.predicted_class.title()} | {result.confidence:.1%}", (22, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {fps:.1f} | Inference: {result.inference_time_ms:.1f} ms", (22, 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            if show_history:
                for offset, item in enumerate(history, start=1):
                    cv2.putText(frame, item, (22, 72 + offset * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 230, 120), 1, cv2.LINE_AA)
            cv2.putText(frame, "Q/Esc: exit | H: history", (22, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (230, 230, 230), 1, cv2.LINE_AA)
            cv2.imshow("Biomedical Waste Segregator", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("h"):
                show_history = not show_history
    finally:
        capture.release()
        cv2.destroyAllWindows()
        predictor.logger.info("Webcam inference stopped safely.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real-time biomedical-waste webcam inference.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    try:
        run_webcam(args.camera, args.model)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        configure_logging(LOG_FILE).getChild("webcam").exception("Webcam inference failed: %s", error)
        raise SystemExit(f"Webcam inference failed: {error}") from error


if __name__ == "__main__":
    main()
