"""Serial transport for legacy numeric commands and newline-delimited JSON."""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass
import serial
from serial.tools import list_ports

@dataclass(frozen=True)
class SerialSettings:
    """Connection settings for a microcontroller serial interface."""
    port: str
    baudrate: int = 9600
    timeout_seconds: float = 1.0

class SerialController:
    """Own serial lifecycle and avoid retries after an ambiguous command write."""
    VALID_COMMANDS = frozenset({"0", "1", "2", "3"})
    MAX_REQUEST_BYTES = 16384
    def __init__(self, settings: SerialSettings | None = None, logger: logging.Logger | None = None) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger("serial_controller")
        self._connection: serial.Serial | None = None
        self._receive_buffer = bytearray()
    @property
    def connected(self) -> bool:
        return self._connection is not None and self._connection.is_open
    @staticmethod
    def available_ports() -> list[dict[str, str]]:
        return [{"device": p.device, "description": p.description, "hwid": p.hwid} for p in list_ports.comports()]
    def connect(self, settings: SerialSettings | None = None) -> None:
        if settings is not None:
            self.settings = settings
        if self.settings is None:
            raise ValueError("Serial settings are required before connecting.")
        self.disconnect()
        try:
            self._connection = serial.Serial(port=self.settings.port, baudrate=self.settings.baudrate,
                timeout=self.settings.timeout_seconds, write_timeout=self.settings.timeout_seconds)
            self._connection.reset_input_buffer()
            self._receive_buffer.clear()
            self.logger.info("Connected to serial port %s at %s baud", self.settings.port, self.settings.baudrate)
        except serial.SerialException as exc:
            self._connection = None
            raise ConnectionError(f"Could not connect to {self.settings.port}: {exc}") from exc
    def disconnect(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            except serial.SerialException as exc:
                self.logger.warning("Serial disconnect warning: %s", exc)
            finally:
                self._connection = None
        self._receive_buffer.clear()
    def read_request_line(self) -> bytes | None:
        """Return one complete buffered JSONL line without blocking."""
        if not self.connected:
            return None
        assert self._connection is not None
        waiting = int(getattr(self._connection, "in_waiting", 0))
        if waiting:
            self._receive_buffer.extend(self._connection.read(min(waiting, self.MAX_REQUEST_BYTES + 1)))
        end = self._receive_buffer.find(b"\n")
        if end >= 0:
            line = bytes(self._receive_buffer[:end]).rstrip(b"\r")
            del self._receive_buffer[:end + 1]
            if len(line) > self.MAX_REQUEST_BYTES:
                raise ValueError("Request line is too large.")
            return line
        if len(self._receive_buffer) > self.MAX_REQUEST_BYTES:
            self._receive_buffer.clear()
            raise ValueError("Request line is too large.")
        return None
    def send_json_response(self, response: dict[str, object]) -> None:
        """Write one JSONL response once; never retry an ambiguous write."""
        from src.communication.json_protocol import encode_response
        if not self.connected:
            raise ConnectionError("Serial hardware is not connected.")
        assert self._connection is not None
        try:
            self._connection.write(encode_response(response))
            self._connection.flush()
        except (serial.SerialException, OSError) as exc:
            self.disconnect()
            raise ConnectionError(f"Unable to write JSON response: {exc}") from exc
    def send_prediction(self, command: int | str, expect_ack: bool = True, retries: int = 1) -> str | None:
        """Send one legacy command. Retries apply only before any command write."""
        normalized = str(command).strip()
        if normalized not in self.VALID_COMMANDS:
            raise ValueError("Serial command must be exactly 0, 1, 2, or 3.")
        last_error: Exception | None = None
        for attempt in range(max(0, retries) + 1):
            if self.connected:
                break
            try:
                self.connect()
                break
            except (serial.SerialException, OSError, ConnectionError) as exc:
                last_error = exc
                self.logger.warning("Serial connection attempt %s failed: %s", attempt + 1, exc)
                if attempt < max(0, retries):
                    time.sleep(0.2)
        if not self.connected:
            raise ConnectionError(f"Unable to connect before sending command {normalized}: {last_error}") from last_error
        assert self._connection is not None
        try:
            self._connection.write(f"{normalized}\n".encode("ascii"))
            self._connection.flush()
            ack = self._connection.readline().decode("utf-8", errors="replace").strip() if expect_ack else None
            if expect_ack and not ack:
                raise TimeoutError("ACK timeout; command was not resent.")
            self.logger.info("Sent command=%s acknowledgement=%s", normalized, ack or "not requested")
            return ack
        except (serial.SerialException, OSError, TimeoutError) as exc:
            self.disconnect()
            raise ConnectionError(f"Command {normalized} may have been received; it was not resent: {exc}") from exc
