"""JSONL request/response validation for the Arduino capture protocol."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from src.decision.mapper import map_prediction

LABELS_PATH = Path(__file__).resolve().parents[2] / "data" / "labels.json"
MAX_LINE_BYTES = 16384


class ProtocolError(ValueError):
    """Malformed or unsupported hardware request."""


def _labels() -> dict[int, str]:
    return {
        int(key): str(value).strip().lower()
        for key, value in json.loads(LABELS_PATH.read_text(encoding="utf-8")).items()
    }


def parse_request_line(line: str | bytes) -> int | str:
    """Validate one {"command":"capture","object_id":...} JSONL request."""
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
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProtocolError("Request is not valid JSON.") from exc
    if not isinstance(request, dict):
        raise ProtocolError("Request must be a JSON object.")
    if set(request) != {"command", "object_id"}:
        raise ProtocolError("Request must contain only command and object_id.")
    object_id = request.get("object_id")
    if isinstance(object_id, bool) or not (
        isinstance(object_id, int)
        or (isinstance(object_id, str) and object_id.strip() and len(object_id) <= 128)
    ):
        raise ProtocolError("object_id must be an integer or a non-empty string of at most 128 characters.")
    if request.get("command") != "capture":
        raise ProtocolError("Unsupported command.")
    return object_id


def _validated_confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError("Prediction confidence is invalid.")
    try:
        confidence = float(value)
    except (OverflowError, ValueError) as exc:
        raise ProtocolError("Prediction confidence is invalid.") from exc
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ProtocolError("Prediction confidence is invalid.")
    return confidence


def prediction_response(object_id: int | str, prediction: Any) -> dict[str, object]:
    """Build the exact Arduino response from a genuine model prediction."""
    labels = _labels()
    index = getattr(prediction, "class_index", None)
    label = getattr(prediction, "predicted_class", None)
    confidence = _validated_confidence(getattr(prediction, "confidence", None))
    if isinstance(index, bool) or not isinstance(index, int) or labels.get(index) != label:
        raise ProtocolError("Prediction does not match configured model labels.")
    if label not in labels.values():
        raise ProtocolError("Prediction label is unsupported.")
    try:
        category = map_prediction(label)
    except ValueError as exc:
        raise ProtocolError("Prediction label has no valid waste category.") from exc
    if category != index:
        raise ProtocolError("Prediction category does not match its model index.")
    return {
        "status": "success",
        "object_id": object_id,
        "category": category,
        "label": label,
        "confidence": round(confidence, 6),
    }


def failed_response(object_id: int | str) -> dict[str, object]:
    """Return the required failure structure for a valid request ID."""
    return {
        "status": "failed",
        "object_id": object_id,
        "category": 0,
        "label": "unknown",
        "confidence": 0.0,
    }


def validate_response(response: dict[str, object]) -> None:
    """Reject invalid outbound response shapes, types, and category values."""
    expected = {"status", "object_id", "category", "label", "confidence"}
    if not isinstance(response, dict) or set(response) != expected:
        raise ProtocolError("Response must contain exactly the five protocol fields.")
    object_id = response["object_id"]
    if isinstance(object_id, bool) or not (
        isinstance(object_id, int)
        or (isinstance(object_id, str) and object_id.strip() and len(object_id) <= 128)
    ):
        raise ProtocolError("Response object_id is invalid.")
    category = response["category"]
    if isinstance(category, bool) or not isinstance(category, int) or not 0 <= category <= 3:
        raise ProtocolError("Response category must be an integer from 0 through 3.")
    confidence = _validated_confidence(response["confidence"])
    status, label = response["status"], response["label"]
    if status == "success":
        labels = _labels()
        if not isinstance(label, str) or labels.get(category) != label:
            raise ProtocolError("Successful response label does not match its category.")
        if map_prediction(label) != category:
            raise ProtocolError("Successful response category does not match the waste mapper.")
    elif status == "failed":
        if category != 0 or label != "unknown" or confidence != 0.0:
            raise ProtocolError("Failed response must use category 0, unknown, and confidence 0.0.")
    else:
        raise ProtocolError("Response status must be success or failed.")


def encode_response(response: dict[str, object]) -> bytes:
    """Validate and encode one compact UTF-8 JSON line."""
    validate_response(response)
    return (
        json.dumps(response, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
