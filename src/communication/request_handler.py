"""Request handling for one validated Arduino capture line."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.communication.json_protocol import (
    ProtocolError, failed_response, parse_request_line, prediction_response,
)


def handle_capture_line(
    line: bytes,
    frame_provider: Callable[[], Any],
    predictor: Any,
) -> tuple[dict[str, object] | None, Any | None, str | None]:
    """Handle one line. Invalid requests are ignored; valid IDs always get one result."""
    try:
        object_id = parse_request_line(line)
    except ProtocolError as exc:
        return None, None, str(exc)
    try:
        frame = frame_provider()
        if frame is None:
            raise ValueError("Camera frame is not available.")
        prediction = predictor.predict_array(frame)
        return prediction_response(object_id, prediction), prediction, None
    except Exception as exc:
        return failed_response(object_id), None, str(exc)
