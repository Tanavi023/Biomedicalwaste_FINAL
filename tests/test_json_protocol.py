"""Tests for the Arduino JSONL request and response contract."""
import json
import math
import unittest
from types import SimpleNamespace

from src.communication.json_protocol import (
    PROTOCOL_LABELS,
    ProtocolError,
    encode_response,
    failed_response,
    parse_request_line,
    prediction_response,
    validate_response,
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
            "label": "metal", "confidence": 0.95,
        })
        payload = encode_response(response)
        self.assertTrue(payload.endswith(b"\n"))
        self.assertEqual(payload.count(b"\n"), 1)
        self.assertEqual(json.loads(payload), response)

    def test_outbound_response_validation_rejects_wrong_types_and_ranges(self):
        invalid = (
            {"status": "success", "object_id": 1, "category": 5, "label": "metal", "confidence": 0.9},
            {"status": "success", "object_id": True, "category": 2, "label": "metal", "confidence": 0.9},
            {"status": "success", "object_id": 1, "category": 2, "label": "metal", "confidence": 1.1},
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

    def test_category_to_label_mapping(self):
        expected_protocol_labels = {0: "yellow", 1: "red", 2: "metal", 3: "white"}
        self.assertEqual(PROTOCOL_LABELS, expected_protocol_labels)

        model_classes = {
            0: "yellow",
            1: "red",
            2: "blue",
            3: "white",
        }
        for class_idx, model_label in model_classes.items():
            prediction = SimpleNamespace(predicted_class=model_label, class_index=class_idx, confidence=0.88)
            resp = prediction_response(42, prediction)
            expected_label = expected_protocol_labels[class_idx]
            self.assertEqual(resp["category"], class_idx)
            self.assertEqual(resp["label"], expected_label)
            self.assertEqual(resp["status"], "success")
            self.assertEqual(resp["object_id"], 42)
            self.assertEqual(resp["confidence"], 0.88)
            validate_response(resp)

    def test_invalid_labels_rejected(self):
        mismatched_labels = (
            {"status": "success", "object_id": 1, "category": 2, "label": "blue", "confidence": 0.9},
            {"status": "success", "object_id": 1, "category": 0, "label": "red", "confidence": 0.9},
            {"status": "success", "object_id": 1, "category": 1, "label": "yellow", "confidence": 0.9},
            {"status": "success", "object_id": 1, "category": 3, "label": "metal", "confidence": 0.9},
            {"status": "success", "object_id": 1, "category": 2, "label": "glass", "confidence": 0.9},
            {"status": "success", "object_id": 1, "category": 2, "label": "unknown", "confidence": 0.9},
            {"status": "success", "object_id": 1, "category": 2, "label": "", "confidence": 0.9},
            {"status": "success", "object_id": 1, "category": 2, "label": None, "confidence": 0.9},
            {"status": "success", "object_id": 1, "category": 2, "label": 2, "confidence": 0.9},
        )
        for resp in mismatched_labels:
            with self.subTest(resp=resp), self.assertRaises(ProtocolError):
                validate_response(resp)

    def test_failed_responses(self):
        resp_int = failed_response(5)
        self.assertEqual(resp_int, {
            "status": "failed", "object_id": 5, "category": 0,
            "label": "unknown", "confidence": 0.0,
        })
        validate_response(resp_int)
        payload_int = encode_response(resp_int)
        self.assertEqual(json.loads(payload_int), resp_int)

        resp_str = failed_response("obj-101")
        self.assertEqual(resp_str, {
            "status": "failed", "object_id": "obj-101", "category": 0,
            "label": "unknown", "confidence": 0.0,
        })
        validate_response(resp_str)
        payload_str = encode_response(resp_str)
        self.assertEqual(json.loads(payload_str), resp_str)

        invalid_failed = (
            {"status": "failed", "object_id": 1, "category": 1, "label": "unknown", "confidence": 0.0},
            {"status": "failed", "object_id": 1, "category": 0, "label": "yellow", "confidence": 0.0},
            {"status": "failed", "object_id": 1, "category": 0, "label": "unknown", "confidence": 0.5},
            {"status": "failed", "object_id": True, "category": 0, "label": "unknown", "confidence": 0.0},
            {"status": "failed", "object_id": 1, "category": 0, "label": "unknown"},
            {"status": "failed", "object_id": 1, "category": 0, "label": "unknown", "confidence": 0.0, "extra": 1},
        )
        for resp in invalid_failed:
            with self.subTest(resp=resp), self.assertRaises(ProtocolError):
                validate_response(resp)

    def test_newline_termination(self):
        prediction = SimpleNamespace(predicted_class="blue", class_index=2, confidence=0.9)
        success_resp = prediction_response(1, prediction)
        success_payload = encode_response(success_resp)
        self.assertTrue(success_payload.endswith(b"\n"))
        self.assertFalse(success_payload.endswith(b"\r\n"))
        self.assertEqual(success_payload.count(b"\n"), 1)

        fail_resp = failed_response(1)
        fail_payload = encode_response(fail_resp)
        self.assertTrue(fail_payload.endswith(b"\n"))
        self.assertFalse(fail_payload.endswith(b"\r\n"))
        self.assertEqual(fail_payload.count(b"\n"), 1)


if __name__ == "__main__":
    unittest.main()
