"""Mocked tests for serial request framing and safe retry behavior."""
import unittest
from unittest.mock import MagicMock
from src.communication.serial_controller import SerialController

class SerialControllerTests(unittest.TestCase):
    def test_invalid_command_is_rejected_before_io(self):
        with self.assertRaises(ValueError):
            SerialController().send_prediction(8)

    def test_send_reads_acknowledgement_once(self):
        controller = SerialController()
        connection = MagicMock()
        connection.is_open = True
        connection.readline.return_value = b"ACK:2\n"
        controller._connection = connection
        self.assertEqual(controller.send_prediction(2), "ACK:2")
        connection.write.assert_called_once_with(b"2\n")

    def test_ack_timeout_does_not_retry_command_write(self):
        controller = SerialController()
        connection = MagicMock()
        connection.is_open = True
        connection.readline.return_value = b""
        controller._connection = connection
        with self.assertRaises(ConnectionError):
            controller.send_prediction(2, retries=3)
        connection.write.assert_called_once_with(b"2\n")

    def test_reads_only_complete_request_lines(self):
        controller = SerialController()
        connection = MagicMock()
        connection.is_open = True
        connection.in_waiting = 8
        connection.read.return_value = b'{"id":1}'
        controller._connection = connection
        self.assertIsNone(controller.read_request_line())
        connection.in_waiting = 1
        connection.read.return_value = b"\n"
        self.assertEqual(controller.read_request_line(), b'{"id":1}')

if __name__ == "__main__":
    unittest.main()
