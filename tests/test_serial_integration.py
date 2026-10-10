"""Mocked serial integration tests for Arduino capture protocol and request processing."""
import json
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import serial

from src.communication.json_protocol import (
    CONFIRMED_PROTOCOL_CATEGORIES,
    PENDING_PROTOCOL_CATEGORIES,
    PROTOCOL_LABELS,
    ProtocolError,
    encode_response,
    failed_response,
    parse_request_line,
    prediction_response,
    validate_response,
)
from src.communication.request_handler import handle_capture_line
from src.communication.serial_controller import SerialController, SerialSettings


class MockSerialConnection:
    """In-memory mock for serial.Serial simulating duplex wire communication."""

    def __init__(self, incoming_bytes: bytes = b"") -> None:
        self._incoming = bytearray(incoming_bytes)
        self._outgoing = bytearray()
        self.is_open = True
        self.write_timeout = 0.1
        self.timeout = 0.1

    @property
    def in_waiting(self) -> int:
        return len(self._incoming)

    def read(self, size: int = 1) -> bytes:
        if not self.is_open:
            raise serial.SerialException("Port is closed.")
        chunk = bytes(self._incoming[:size])
        del self._incoming[:size]
        return chunk

    def write(self, data: bytes) -> int:
        if not self.is_open:
            raise serial.SerialException("Port is closed.")
        self._outgoing.extend(data)
        return len(data)

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False

    def reset_input_buffer(self) -> None:
        self._incoming.clear()

    def get_written_bytes(self) -> bytes:
        return bytes(self._outgoing)

    def get_written_lines(self) -> list[bytes]:
        return [line for line in self._outgoing.split(b"\n") if line]


class SerialIntegrationTests(unittest.TestCase):
    """End-to-end mocked hardware exchange testing without physical Uno."""

    def create_controller(self, incoming: bytes = b"") -> tuple[SerialController, MockSerialConnection]:
        mock_conn = MockSerialConnection(incoming)
        controller = SerialController(SerialSettings(port="COM_MOCK", baudrate=9600))
        controller._connection = mock_conn
        return controller, mock_conn

    def test_valid_capture_numeric_id_success(self):
        """Valid capture with numeric ID triggers inference and outputs confirmed label."""
        controller, mock_conn = self.create_controller(b'{"command":"capture","object_id":1}\n')
        line = controller.read_request_line()
        self.assertEqual(line, b'{"command":"capture","object_id":1}')

        prediction = SimpleNamespace(predicted_class="blue", class_index=2, confidence=0.96)
        predictor = Mock()
        predictor.predict_array.return_value = prediction

        response, result, error = handle_capture_line(line, lambda: "frame-data", predictor)
        self.assertIsNone(error)
        self.assertIsNotNone(response)
        self.assertEqual(response["status"], "success")
        self.assertEqual(response["object_id"], 1)
        self.assertEqual(response["category"], 2)
        self.assertEqual(response["label"], "metal")
        self.assertIn(2, CONFIRMED_PROTOCOL_CATEGORIES)

        controller.send_json_response(response)
        written = mock_conn.get_written_bytes()
        self.assertTrue(written.endswith(b"\n"))
        self.assertFalse(written.endswith(b"\r\n"))
        self.assertEqual(written.count(b"\n"), 1)
        parsed = json.loads(written.decode("utf-8"))
        self.assertEqual(parsed, {
            "status": "success", "object_id": 1, "category": 2,
            "label": "metal", "confidence": 0.96,
        })
        # Verify compact JSON serialization (no whitespace after separators)
        self.assertNotIn(b": ", written)
        self.assertNotIn(b", ", written)

    def test_valid_capture_string_id_success(self):
        """Valid capture with string object ID preserves ID and pending label."""
        controller, mock_conn = self.create_controller(b'{"command":"capture","object_id":"batch-99"}\n')
        line = controller.read_request_line()
        self.assertEqual(line, b'{"command":"capture","object_id":"batch-99"}')

        prediction = SimpleNamespace(predicted_class="yellow", class_index=0, confidence=0.87)
        predictor = Mock()
        predictor.predict_array.return_value = prediction

        response, result, error = handle_capture_line(line, lambda: "frame", predictor)
        self.assertIsNone(error)
        self.assertEqual(response["object_id"], "batch-99")
        self.assertEqual(response["category"], 0)
        self.assertEqual(response["label"], "yellow")
        self.assertIn(0, PENDING_PROTOCOL_CATEGORIES)

        controller.send_json_response(response)
        lines = mock_conn.get_written_lines()
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0].decode("utf-8"))
        self.assertEqual(parsed["object_id"], "batch-99")

    def test_camera_failure_returns_prescribed_failed_response(self):
        """Camera unavailability returns status=failed with category 0 and label unknown."""
        controller, mock_conn = self.create_controller(b'{"command":"capture","object_id":42}\n')
        line = controller.read_request_line()

        predictor = Mock()
        response, result, error = handle_capture_line(line, lambda: None, predictor)
        self.assertIsNotNone(error)
        self.assertIsNone(result)
        self.assertEqual(response, {
            "status": "failed", "object_id": 42, "category": 0,
            "label": "unknown", "confidence": 0.0,
        })
        predictor.predict_array.assert_not_called()

        controller.send_json_response(response)
        lines = mock_conn.get_written_lines()
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0].decode("utf-8"))
        self.assertEqual(parsed["status"], "failed")
        self.assertEqual(parsed["object_id"], 42)

    def test_inference_exception_returns_prescribed_failed_response(self):
        """Inference runtime exception safely returns failed response with the request ID."""
        controller, mock_conn = self.create_controller(b'{"command":"capture","object_id":77}\n')
        line = controller.read_request_line()

        predictor = Mock()
        predictor.predict_array.side_effect = RuntimeError("Inference execution crashed")

        response, result, error = handle_capture_line(line, lambda: "frame", predictor)
        self.assertIn("Inference execution crashed", error)
        self.assertEqual(response["status"], "failed")
        self.assertEqual(response["object_id"], 77)

        controller.send_json_response(response)
        lines = mock_conn.get_written_lines()
        self.assertEqual(len(lines), 1)

    def test_multiple_consecutive_requests_with_respective_ids(self):
        """Multiple back-to-back requests are processed in order with exactly one response each."""
        input_data = (
            b'{"command":"capture","object_id":10}\n'
            b'{"command":"capture","object_id":20}\n'
            b'{"command":"capture","object_id":30}\n'
        )
        controller, mock_conn = self.create_controller(input_data)

        predictions = [
            SimpleNamespace(predicted_class="red", class_index=1, confidence=0.91),
            SimpleNamespace(predicted_class="blue", class_index=2, confidence=0.94),
            SimpleNamespace(predicted_class="white", class_index=3, confidence=0.88),
        ]
        predictor = Mock()
        predictor.predict_array.side_effect = predictions

        for expected_id, expected_cat, expected_label in (
            (10, 1, "red"),
            (20, 2, "metal"),
            (30, 3, "white"),
        ):
            line = controller.read_request_line()
            self.assertIsNotNone(line)
            response, result, error = handle_capture_line(line, lambda: "frame", predictor)
            self.assertIsNone(error)
            self.assertEqual(response["object_id"], expected_id)
            self.assertEqual(response["category"], expected_cat)
            self.assertEqual(response["label"], expected_label)
            controller.send_json_response(response)

        # No more buffered lines
        self.assertIsNone(controller.read_request_line())

        lines = mock_conn.get_written_lines()
        self.assertEqual(len(lines), 3)
        ids = [json.loads(l.decode("utf-8"))["object_id"] for l in lines]
        self.assertEqual(ids, [10, 20, 30])
        labels = [json.loads(l.decode("utf-8"))["label"] for l in lines]
        self.assertEqual(labels, ["red", "metal", "white"])

    def test_malformed_json_and_unsupported_commands_yield_no_response(self):
        """Invalid requests are ignored with zero responses and no inference executed."""
        invalid_inputs = [
            b"this is not json\n",
            b'{"command":"status","object_id":1}\n',
            b'{"command":"capture"}\n',
            b'{"command":"capture","object_id":true}\n',
            b'{"command":"capture","object_id":""}\n',
            b'{"command":"capture","object_id":1.5}\n',
            b'{"command":"capture","object_id":1,"extra":"field"}\n',
        ]
        for bad_line in invalid_inputs:
            with self.subTest(bad_line=bad_line):
                controller, mock_conn = self.create_controller(bad_line)
                line = controller.read_request_line()
                predictor = Mock()
                response, result, error = handle_capture_line(line, lambda: "frame", predictor)
                self.assertIsNone(response)
                self.assertIsNone(result)
                self.assertIsNotNone(error)
                predictor.predict_array.assert_not_called()
                # Ensure no bytes were written to serial
                self.assertEqual(mock_conn.get_written_bytes(), b"")

    def test_invalid_labels_and_response_structures_rejected(self):
        """Protocol validation rejects mismatched labels, invalid statuses, and corrupted shapes."""
        invalid_responses = [
            # Category 2 with old model label 'blue' instead of protocol label 'metal'
            {"status": "success", "object_id": 1, "category": 2, "label": "blue", "confidence": 0.95},
            # Category 0 with label 'metal'
            {"status": "success", "object_id": 1, "category": 0, "label": "metal", "confidence": 0.95},
            # Category 1 with label 'white'
            {"status": "success", "object_id": 1, "category": 1, "label": "white", "confidence": 0.95},
            # Category 3 with label 'glass'
            {"status": "success", "object_id": 1, "category": 3, "label": "glass", "confidence": 0.95},
            # Out of bounds category
            {"status": "success", "object_id": 1, "category": 4, "label": "metal", "confidence": 0.95},
            # Invalid status
            {"status": "pending", "object_id": 1, "category": 2, "label": "metal", "confidence": 0.95},
            # Confidence out of range
            {"status": "success", "object_id": 1, "category": 2, "label": "metal", "confidence": 1.5},
            # Failed response with wrong fields
            {"status": "failed", "object_id": 1, "category": 2, "label": "metal", "confidence": 0.0},
        ]
        for invalid_resp in invalid_responses:
            with self.subTest(invalid_resp=invalid_resp):
                with self.assertRaises(ProtocolError):
                    validate_response(invalid_resp)
                with self.assertRaises(ProtocolError):
                    encode_response(invalid_resp)

    def test_serial_read_errors_write_errors_and_connection_loss(self):
        """Serial hardware errors disconnect the controller and raise ConnectionError."""
        # Read error
        controller, mock_conn = self.create_controller(b"partial")
        mock_conn.read = Mock(side_effect=serial.SerialException("device detached"))
        with self.assertRaises(ConnectionError):
            controller.read_request_line()
        self.assertFalse(controller.connected)

        # Write error
        controller, mock_conn = self.create_controller()
        mock_conn.write = Mock(side_effect=OSError("write timeout"))
        with self.assertRaises(ConnectionError):
            controller.send_json_response({
                "status": "failed", "object_id": 1, "category": 0,
                "label": "unknown", "confidence": 0.0,
            })
        self.assertFalse(controller.connected)

        # Write while disconnected
        controller.disconnect()
        with self.assertRaises(ConnectionError):
            controller.send_json_response({
                "status": "failed", "object_id": 1, "category": 0,
                "label": "unknown", "confidence": 0.0,
            })

    def test_request_processor_handles_invalid_success_with_failed_fallback(self):
        """An invalid success response is caught, logged, and replaced with failed_response."""
        controller, mock_conn = self.create_controller()
        request_queue = queue.Queue()
        ui_events = queue.Queue()
        audit_log = Mock()
        listener_stop = threading.Event()

        # Simulate a handler returning an invalid success response (mismatched label)
        invalid_success = {
            "status": "success", "object_id": 55, "category": 2,
            "label": "blue", "confidence": 0.9,  # 'blue' is invalid; protocol requires 'metal'
        }

        # Put line in queue, then poison pill None
        request_queue.put(b'{"command":"capture","object_id":55}')
        request_queue.put(None)

        # Run processor loop logic
        while not listener_stop.is_set():
            try:
                line = request_queue.get(timeout=0.1)
            except queue.Empty:
                break
            if line is None:
                break

            response = invalid_success
            object_id = response.get("object_id")

            try:
                controller.send_json_response(response)
            except ProtocolError as exc:
                ui_events.put(("status", f"Invalid response: {exc}"))
                if response.get("status") == "success":
                    audit_log.log_failure(object_id, f"Invalid success response: {exc}")
                    fallback = failed_response(object_id)
                    controller.send_json_response(fallback)
                    ui_events.put(("serial_exchange_verified", None))

        # Exactly ONE response was written to serial (the fallback failed response)
        lines = mock_conn.get_written_lines()
        self.assertEqual(len(lines), 1)
        parsed = json.loads(lines[0].decode("utf-8"))
        self.assertEqual(parsed["status"], "failed")
        self.assertEqual(parsed["object_id"], 55)
        self.assertEqual(parsed["category"], 0)
        self.assertEqual(parsed["label"], "unknown")
        audit_log.log_failure.assert_called_once()

    def test_request_processor_stops_cleanly_on_connection_error_without_duplicate_reply(self):
        """Serial write failure stops the listener cleanly without duplicating replies."""
        controller, mock_conn = self.create_controller()
        # Simulate connection loss on write
        mock_conn.write = Mock(side_effect=ConnectionError("USB link dropped"))
        listener_stop = threading.Event()
        ui_events = queue.Queue()

        valid_response = {
            "status": "success", "object_id": 99, "category": 2,
            "label": "metal", "confidence": 0.95,
        }

        # Process single response
        try:
            controller.send_json_response(valid_response)
        except ConnectionError as exc:
            ui_events.put(("connection_error", str(exc)))
            listener_stop.set()

        self.assertTrue(listener_stop.is_set())
        event_name, event_val = ui_events.get_nowait()
        self.assertEqual(event_name, "connection_error")
        # No duplicate reply was written
        self.assertEqual(mock_conn.get_written_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
