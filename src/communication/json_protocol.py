"""JSONL request/response validation for the hardware link."""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Any

LABELS_PATH = Path(__file__).resolve().parents[2] / "data" / "labels.json"
MAX_LINE_BYTES = 16384

class ProtocolError(ValueError):
    """Malformed or unsupported hardware request."""

def _labels() -> dict[int, str]:
    return {int(k): str(v).strip().lower() for k, v in json.loads(LABELS_PATH.read_text(encoding="utf-8")).items()}

def parse_request_line(line: str | bytes) -> str:
    """Parse one JSONL request: {"object_id":"id","action":"classify"}."""
    if isinstance(line, bytes):
        if len(line) > MAX_LINE_BYTES:
            raise ProtocolError("Request line is too large.")
        try:
            line = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("Request must be UTF-8 JSON.") from exc
    if not isinstance(line, str) or len(line.encode("utf-8")) > MAX_LINE_BYTES:
        raise ProtocolError("Request line is invalid or too large.")
    try:
        request = json.loads(line)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProtocolError("Request is not valid JSON.") from exc
    if not isinstance(request, dict):
        raise ProtocolError("Request must be a JSON object.")
    object_id = request.get("object_id")
    if not isinstance(object_id, str) or not object_id.strip() or len(object_id) > 128:
        raise ProtocolError("object_id must be a non-empty string of at most 128 characters.")
    if request.get("action", "classify") != "classify":
        raise ProtocolError("Unsupported action.")
    return object_id.strip()

def prediction_response(object_id: str, prediction: Any) -> dict[str, object]:
    """Build a response matching the current four-class model labels."""
    labels = _labels()
    index, label = getattr(prediction, "class_index", None), getattr(prediction, "predicted_class", None)
    confidence, elapsed = getattr(prediction, "confidence", None), getattr(prediction, "inference_time_ms", None)
    if isinstance(index, bool) or not isinstance(index, int) or labels.get(index) != label:
        raise ProtocolError("Prediction does not match configured model labels.")
    if label not in {"yellow", "red", "blue", "white"}:
        raise ProtocolError("Prediction label is unsupported.")
    if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ProtocolError("Prediction confidence is invalid.")
    if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
        raise ProtocolError("Prediction time is invalid.")
    return {"object_id": object_id, "status": "ok", "predicted_class": label,
            "class_index": index, "confidence": round(float(confidence), 6),
            "inference_time_ms": round(float(elapsed), 3),
            "bin": {"yellow": 0, "red": 1, "blue": 2, "white": 3}[label]}

def error_response(object_id: str | None, message: str) -> dict[str, object]:
    safe_id = object_id if isinstance(object_id, str) and object_id.strip() else None
    return {"object_id": safe_id, "status": "error",
            "error": str(message).replace("\r", " ").replace("\n", " ")[:200]}

def encode_response(response: dict[str, object]) -> bytes:
    """Encode one compact UTF-8 JSON response, terminated by a newline."""
    return (json.dumps(response, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
