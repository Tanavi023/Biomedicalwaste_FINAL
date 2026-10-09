"""Mocked tests for serial framing and newline-terminated output."""
import unittest
from unittest.mock import MagicMock

import serial

from src.communication.serial_controller import SerialController, SerialSettings


class SerialControllerTests(unittest.TestCase):
    def connected_controller(self, incoming=b""):
        controller = SerialController()
        connection = MagicMock()
        connection.is_open = True
        connection.in_waiting = len(incoming)
        connection.read.return_value = incoming
        controller._connection = connection
        return controller, connection

    def test_nondefault_baudrate_is_configurable(self):
        settings = SerialSettings(port="COM7", baudrate=115200, timeout_seconds=0.1)
        self.assertEqual(settings.port, "COM7")
        self.assertEqual(settings.baudrate, 115200)
        self.assertEqual(settings.timeout_seconds, 0.1)

    def test_invalid_legacy_command_is_rejected_before_io(self):
        with self.assertRaises(ValueError):
            SerialController().send_prediction(8)

    def test_legacy_send_reads_acknowledgement_once(self):
        controller, connection = self.connected_controller()
        connection.readline.return_value = b"ACK:2\n"
        self.assertEqual(controller.send_prediction(2), "ACK:2")
        connection.write.assert_called_once_with(b"2\n")

    def test_ack_timeout_does_not_retry_command_write(self):
        controller, connection = self.connected_controller()
        connection.readline.return_value = b""
        with self.assertRaises(ConnectionError):
            controller.send_prediction(2, retries=3)
        connection.write.assert_called_once_with(b"2\n")

    def test_reads_complete_lines_and_multiple_buffered_requests(self):
        data = b'{"command":"capture","object_id":1}\n{"command":"capture","object_id":2}\n'
        controller, connection = self.connected_controller(data)
        connection.in_waiting = len(data)
        connection.read.return_value = data
        self.assertEqual(controller.read_request_line(), b'{"command":"capture","object_id":1}')
        self.assertEqual(controller.read_request_line(), b'{"command":"capture","object_id":2}')

    def test_reads_partial_line_only_after_newline_arrives(self):
        controller, connection = self.connected_controller(b'{"command":')
        self.assertIsNone(controller.read_request_line())
        connection.in_waiting = 10
        connection.read.return_value = b'"capture"}\n'
        self.assertEqual(controller.read_request_line(), b'{"command":"capture"}')

    def test_serial_read_failure_marks_connection_disconnected(self):
        controller, connection = self.connected_controller(b"x")
        connection.read.side_effect = serial.SerialException("device removed")
        with self.assertRaises(ConnectionError):
            controller.read_request_line()
        self.assertIsNone(controller._connection)
        connection.close.assert_called_once()
    def test_serial_write_failure_marks_connection_disconnected(self):
        controller, connection = self.connected_controller()
        connection.write.side_effect = OSError("device removed")
        with self.assertRaises(ConnectionError):
            controller.send_json_response({"status": "failed", "object_id": 1, "category": 0, "label": "unknown", "confidence": 0.0})
        self.assertIsNone(controller._connection)
        connection.close.assert_called_once()
    def test_writes_one_json_line_with_newline(self):
        controller, connection = self.connected_controller()
        response = {"status": "failed", "object_id": 1, "category": 0, "label": "unknown", "confidence": 0.0}
        controller.send_json_response(response)
        connection.write.assert_called_once_with(
            b'{"status":"failed","object_id":1,"category":0,"label":"unknown","confidence":0.0}\n'
        )
        connection.flush.assert_called_once()


if __name__ == "__main__":
    unittest.main()
