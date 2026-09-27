"""Tests for the JSONL hardware protocol."""
import unittest
from types import SimpleNamespace

from src.communication.json_protocol import (
    ProtocolError, encode_response, error_response, parse_request_line, prediction_response,
)

class JsonProtocolTests(unittest.TestCase):
    def test_parses_one_request(self):
        self.assertEqual(parse_request_line(b'{"object_id":"obj-17","action":"classify"}\r\n'), "obj-17")

    def test_rejects_bad_requests(self):
        for line in (b"not-json", b"{}", b'{"object_id":"x","action":"open"}'):
            with self.subTest(line=line), self.assertRaises(ProtocolError):
                parse_request_line(line)

    def test_actual_blue_label_response_and_newline(self):
        prediction = SimpleNamespace(predicted_class="blue", class_index=2, confidence=0.95, inference_time_ms=8.2)
        response = prediction_response("glass-1", prediction)
        self.assertEqual(response["predicted_class"], "blue")
        self.assertEqual(response["bin"], 2)
        encoded = encode_response(response)
        self.assertTrue(encoded.endswith(b"\n"))
        self.assertEqual(encoded.count(b"\n"), 1)

    def test_rejects_non_model_output_alias(self):
        prediction = SimpleNamespace(predicted_class="metal", class_index=2, confidence=0.95, inference_time_ms=8.2)
        with self.assertRaises(ProtocolError):
            prediction_response("metal-1", prediction)

    def test_error_response_is_bounded_and_safe(self):
        response = error_response(None, "bad\nrequest" * 100)
        self.assertIsNone(response["object_id"])
        self.assertLessEqual(len(response["error"]), 200)

if __name__ == "__main__":
    unittest.main()
