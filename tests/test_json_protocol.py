"""Tests for the Arduino JSONL request and response contract."""
import json
import math
import unittest
from types import SimpleNamespace

from src.communication.json_protocol import (
    ProtocolError, encode_response, failed_response, parse_request_line, prediction_response,
)


class JsonProtocolTests(unittest.TestCase):
    def test_parses_capture_and_preserves_numeric_or_string_id(self):
        self.assertEqual(parse_request_line(b'{"command":"capture","object_id":1}\r\n'), 1)
        self.assertEqual(parse_request_line('{"command":"capture","object_id":" obj-2 "}'), " obj-2 ")

    def test_invalid_json_object_command_and_ids_are_rejected(self):
        requests = (
            b"not-json", b"[]", b'{"command":"capture"}',
            b'{"command":"capture","object_id":true}',
            b'{"command":"capture","object_id":1.5}',
            b'{"command":"capture","object_id":""}',
            b'{"command":"capture","object_id":"x","extra":2}',
            b'{"command":"other","object_id":1}',
        )
        for request in requests:
            with self.subTest(request=request), self.assertRaises(ProtocolError):
                parse_request_line(request)

    def test_success_response_has_exact_fields_and_newline(self):
        prediction = SimpleNamespace(predicted_class="blue", class_index=2, confidence=0.95)
        response = prediction_response(1, prediction)
        self.assertEqual(response, {
            "status": "success", "object_id": 1, "category": 2,
            "label": "blue", "confidence": 0.95,
        })
        payload = encode_response(response)
        self.assertTrue(payload.endswith(b"\n"))
        self.assertEqual(payload.count(b"\n"), 1)
        self.assertEqual(json.loads(payload), response)

    def test_outbound_response_validation_rejects_wrong_types_and_ranges(self):
        invalid = (
            {"status": "success", "object_id": 1, "category": 5, "label": "blue", "confidence": 0.9},
            {"status": "success", "object_id": True, "category": 2, "label": "blue", "confidence": 0.9},
            {"status": "success", "object_id": 1, "category": 2, "label": "blue", "confidence": 1.1},
            {"status": "success", "object_id": 1, "category": 2, "label": "red", "confidence": 0.9},
            {"status": "failed", "object_id": 1, "category": 2, "label": "unknown", "confidence": 0.0},
        )
        for response in invalid:
            with self.subTest(response=response), self.assertRaises(ProtocolError):
                encode_response(response)
    def test_failed_response_has_required_shape(self):
        self.assertEqual(failed_response(1), {
            "status": "failed", "object_id": 1, "category": 0,
            "label": "unknown", "confidence": 0.0,
        })

    def test_confidence_and_prediction_label_are_validated(self):
        for value in (-0.1, 1.1, float("nan"), float("inf"), True, "0.8"):
            prediction = SimpleNamespace(predicted_class="blue", class_index=2, confidence=value)
            with self.subTest(confidence=value), self.assertRaises(ProtocolError):
                prediction_response(1, prediction)
        wrong = SimpleNamespace(predicted_class="metal", class_index=2, confidence=0.95)
        with self.assertRaises(ProtocolError):
            prediction_response(1, wrong)


if __name__ == "__main__":
    unittest.main()
