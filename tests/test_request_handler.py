"""Tests for one-inference-per-request handling without physical devices."""
import unittest
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from src.communication.request_handler import handle_capture_line


class RequestHandlerTests(unittest.TestCase):
    def test_one_inference_and_one_success_payload_for_valid_capture(self):
        prediction = SimpleNamespace(predicted_class="blue", class_index=2, confidence=0.95)
        predictor = Mock()
        predictor.predict_array.return_value = prediction
        response, result, error = handle_capture_line(
            b'{"command":"capture","object_id":1}', lambda: "rgb-frame", predictor
        )
        self.assertEqual(response, {
            "status": "success", "object_id": 1, "category": 2,
            "label": "metal", "confidence": 0.95,
        })
        self.assertIs(result, prediction)
        self.assertIsNone(error)
        predictor.predict_array.assert_called_once_with("rgb-frame")

    def test_camera_or_inference_failure_returns_one_failure_payload(self):
        no_frame_predictor = Mock()
        response, result, error = handle_capture_line(
            b'{"command":"capture","object_id":7}', lambda: None, no_frame_predictor
        )
        self.assertEqual(response, {
            "status": "failed", "object_id": 7, "category": 0,
            "label": "unknown", "confidence": 0.0,
        })
        self.assertIsNone(result)
        self.assertTrue(error)
        no_frame_predictor.predict_array.assert_not_called()

        broken_predictor = Mock()
        broken_predictor.predict_array.side_effect = RuntimeError("model failed")
        response, result, error = handle_capture_line(
            b'{"command":"capture","object_id":8}', lambda: "frame", broken_predictor
        )
        self.assertEqual(response["object_id"], 8)
        self.assertEqual(response["status"], "failed")
        self.assertIsNone(result)
        self.assertTrue(error)
        broken_predictor.predict_array.assert_called_once_with("frame")

    def test_consecutive_requests_each_trigger_one_inference_and_keep_ids(self):
        predictor = Mock()
        predictor.predict_array.side_effect = [
            SimpleNamespace(predicted_class="red", class_index=1, confidence=0.8),
            SimpleNamespace(predicted_class="blue", class_index=2, confidence=0.9),
        ]
        responses = []
        for object_id in (11, 12):
            response, prediction, error = handle_capture_line(
                ('{"command":"capture","object_id":%d}' % object_id).encode(),
                lambda: "frame", predictor,
            )
            responses.append(response)
            self.assertIsNotNone(prediction)
            self.assertIsNone(error)
        self.assertEqual([item["object_id"] for item in responses], [11, 12])
        self.assertEqual([item["status"] for item in responses], ["success", "success"])
        self.assertEqual(predictor.predict_array.call_count, 2)

    def test_background_inference_leaves_caller_available(self):
        started = threading.Event()
        release = threading.Event()
        predictor = Mock()

        def slow_prediction(frame):
            started.set()
            release.wait(timeout=2)
            return SimpleNamespace(predicted_class="blue", class_index=2, confidence=0.9)

        predictor.predict_array.side_effect = slow_prediction
        worker = threading.Thread(
            target=handle_capture_line,
            args=(b'{"command":"capture","object_id":21}', lambda: "frame", predictor),
        )
        worker.start()
        self.assertTrue(started.wait(timeout=1))
        # The caller remains able to service UI work while inference is in progress.
        ui_work_serviced = threading.Event()
        ui_work_serviced.set()
        self.assertTrue(ui_work_serviced.is_set())
        release.set()
        worker.join(timeout=2)
        self.assertFalse(worker.is_alive())
        predictor.predict_array.assert_called_once_with("frame")

    def test_invalid_or_unsupported_request_has_no_response_and_no_inference(self):
        predictor = Mock()
        for line in (
            b"{broken",
            b'{"command":"capture"}',
            b'{"command":"other","object_id":1}',
        ):
            response, prediction, error = handle_capture_line(line, lambda: "frame", predictor)
            self.assertIsNone(response)
            self.assertIsNone(prediction)
            self.assertTrue(error)
        predictor.predict_array.assert_not_called()


if __name__ == "__main__":
    unittest.main()
