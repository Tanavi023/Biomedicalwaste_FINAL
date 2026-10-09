"""Tkinter dashboard with preview-only camera and background Arduino requests."""
from __future__ import annotations

import queue
import logging
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from src.communication.request_handler import handle_capture_line
from src.communication.serial_controller import SerialController, SerialSettings
from src.decision.mapper import map_prediction
from src.inference.predict_image import DEFAULT_MODEL_PATH, BiomedicalWastePredictor
from src.utils.prediction_logger import DEFAULT_LOG_PATH, PredictionLogger

LOGGER = logging.getLogger(__name__)


class BiomedicalWasteDashboard(tk.Tk):
    """Camera preview is passive; valid serial requests trigger inference."""

    def __init__(self) -> None:
        super().__init__()
        self.title("AI-Driven Biomedical Waste Segregator")
        self.geometry("1080x780")
        self.minsize(900, 680)
        self.configure(bg="#F4F7FB")
        self.predictor = BiomedicalWastePredictor(DEFAULT_MODEL_PATH)
        self.serial = SerialController()
        self.audit_log = PredictionLogger()
        self.capture: cv2.VideoCapture | None = None
        self._camera_stop = threading.Event()
        self._camera_thread: threading.Thread | None = None
        self._manual_thread: threading.Thread | None = None
        self.running = False
        self.count = 0
        self.latest_rgb_frame = None
        self._photo: ImageTk.PhotoImage | None = None
        self._frame_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._listener_stop = threading.Event()
        self._request_queue: queue.Queue[bytes | None] = queue.Queue()
        self._ui_events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._listener_thread: threading.Thread | None = None
        self._processor_thread: threading.Thread | None = None
        self._connected_port = ""
        self._connected_baudrate = 0
        self._serial_exchange_verified = False
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.after(50, self._drain_ui_events)

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), background="#F4F7FB", foreground="#0B2D4D")
        style.configure("Metric.TLabel", font=("Segoe UI", 14, "bold"), background="white", foreground="#0B2D4D")
        header = ttk.Frame(self, padding=(24, 18))
        header.pack(fill="x")
        ttk.Label(header, text="AI-Driven Biomedical Waste Segregator", style="Title.TLabel").pack(side="left")
        self.connection_var = tk.StringVar(value="Serial: disconnected")
        ttk.Label(header, textvariable=self.connection_var).pack(side="right")
        body = ttk.Frame(self, padding=(24, 0, 24, 20))
        body.pack(fill="both", expand=True)
        camera_frame = ttk.LabelFrame(body, text="Live Camera Preview (no automatic inference)", padding=10)
        camera_frame.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.camera_label = ttk.Label(camera_frame, text="Camera stopped", anchor="center")
        self.camera_label.pack(fill="both", expand=True)
        panel = ttk.Frame(body, width=330)
        panel.pack(side="right", fill="y")
        self.prediction_var = tk.StringVar(value="-")
        self.confidence_var = tk.StringVar(value="-")
        self.time_var = tk.StringVar(value="-")
        self.category_var = tk.StringVar(value="-")
        self.counter_var = tk.StringVar(value="0")
        for title, variable in (("Prediction", self.prediction_var), ("Confidence", self.confidence_var),
                                ("Inference time", self.time_var), ("Waste category", self.category_var),
                                ("Prediction counter", self.counter_var)):
            card = ttk.LabelFrame(panel, text=title, padding=10)
            card.pack(fill="x", pady=4)
            ttk.Label(card, textvariable=variable, style="Metric.TLabel").pack(anchor="w")
        controls = ttk.Frame(panel, padding=(0, 12))
        controls.pack(fill="x")
        ttk.Button(controls, text="Start Camera", command=self.start_camera).pack(fill="x", pady=3)
        ttk.Button(controls, text="Stop Camera", command=self.stop_camera).pack(fill="x", pady=3)
        ttk.Button(controls, text="Classify Current Frame", command=self.classify_current_frame).pack(fill="x", pady=3)
        self.port_var = tk.StringVar(value="")
        ttk.Label(controls, text="Arduino COM port").pack(anchor="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.port_var).pack(fill="x", pady=3)
        self.baud_var = tk.StringVar(value="9600")
        ttk.Label(controls, text="Baud rate (must match Arduino firmware)").pack(anchor="w")
        ttk.Entry(controls, textvariable=self.baud_var).pack(fill="x", pady=3)
        ttk.Button(controls, text="Connect Hardware", command=self.connect_hardware).pack(fill="x", pady=3)
        ttk.Button(controls, text="Disconnect Hardware", command=self.disconnect_hardware).pack(fill="x", pady=3)
        ttk.Button(controls, text="Export Logs", command=self.export_logs).pack(fill="x", pady=(10, 3))
        ttk.Button(controls, text="Clear Logs", command=self.clear_logs).pack(fill="x", pady=3)
        self.status_var = tk.StringVar(value="System ready. Model loaded.")
        ttk.Label(self, textvariable=self.status_var, padding=(24, 8), anchor="w").pack(fill="x")

    def start_camera(self) -> None:
        if self.running:
            return
        if self._camera_thread is not None and self._camera_thread.is_alive():
            self.status_var.set("Camera is still stopping; try again shortly.")
            return
        self.running = True
        self._camera_stop.clear()
        self.status_var.set("Starting camera...")
        self._camera_thread = threading.Thread(
            target=self._camera_capture_loop, name="camera-capture", daemon=True
        )
        self._camera_thread.start()
        self._update_frame()

    def _camera_capture_loop(self) -> None:
        camera = None
        try:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                self._ui_events.put(("camera_error", "Could not open the default webcam."))
                return
            self.capture = camera
            self._ui_events.put(("camera_started", None))
            while not self._camera_stop.is_set():
                ok, frame = camera.read()
                if not ok:
                    if not self._camera_stop.is_set():
                        self._ui_events.put(("camera_error", "Camera frame could not be read."))
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self._frame_lock:
                    self.latest_rgb_frame = rgb
        except Exception as exc:
            if not self._camera_stop.is_set():
                self._ui_events.put(("camera_error", f"Camera capture failed: {exc}"))
        finally:
            if camera is not None:
                camera.release()
            if self.capture is camera:
                self.capture = None
            with self._frame_lock:
                self.latest_rgb_frame = None

    def _update_frame(self) -> None:
        if not self.running:
            return
        with self._frame_lock:
            frame = None if self.latest_rgb_frame is None else self.latest_rgb_frame.copy()
        if frame is not None:
            self._photo = ImageTk.PhotoImage(Image.fromarray(frame).resize((640, 480)))
            self.camera_label.configure(image=self._photo, text="")
        self.after(30, self._update_frame)

    def _start_serial_workers(self) -> None:
        self._listener_stop.clear()
        self._request_queue = queue.Queue()
        self._listener_thread = threading.Thread(target=self._serial_listener, name="serial-listener", daemon=True)
        self._processor_thread = threading.Thread(target=self._request_processor, name="serial-classifier", daemon=True)
        self._listener_thread.start()
        self._processor_thread.start()

    def _serial_listener(self) -> None:
        while not self._listener_stop.is_set() and self.serial.connected:
            try:
                line = self.serial.read_request_line()
            except ConnectionError as exc:
                LOGGER.error("Serial listener disconnected: %s", exc)
                self._ui_events.put(("connection_error", str(exc)))
                self._listener_stop.set()
                break
            except ValueError as exc:
                LOGGER.warning("Rejected oversized serial line: %s", exc)
                self._ui_events.put(("status", f"Hardware request rejected: {exc}"))
                continue
            if line is None:
                self._listener_stop.wait(0.01)
            else:
                self._request_queue.put(line)
        self._request_queue.put(None)

    def _request_processor(self) -> None:
        while not self._listener_stop.is_set():
            try:
                line = self._request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                break

            def current_frame():
                with self._frame_lock:
                    return None if self.latest_rgb_frame is None else self.latest_rgb_frame.copy()

            with self._inference_lock:
                response, prediction, error = handle_capture_line(
                    line, current_frame, self.predictor
                )
            if response is None:
                self._ui_events.put(("status", f"Hardware request ignored: {error}"))
                continue

            object_id = response["object_id"]
            if prediction is not None:
                category = int(response["category"])
                try:
                    self.audit_log.log(prediction, category, f"json_request_classified; object_id={response['object_id']}")
                except Exception as exc:
                    self._ui_events.put(("status", f"Prediction log write failed: {exc}"))
                self._ui_events.put(("prediction", (prediction, category)))
            else:
                try:
                    self.audit_log.log_failure(object_id, error or "Classification failed.")
                except Exception as exc:
                    self._ui_events.put(("status", f"Failure log write failed: {exc}"))
                self._ui_events.put(("status", f"Classification failed for request {object_id}: {error}"))

            try:
                self.serial.send_json_response(response)
                self._ui_events.put(("serial_exchange_verified", None))
                self._ui_events.put(("status", f"Processed Arduino request {object_id}."))
            except ConnectionError as exc:
                self._ui_events.put(("connection_error", str(exc)))
                self._listener_stop.set()
                break

    def _drain_ui_events(self) -> None:
        try:
            while True:
                event, value = self._ui_events.get_nowait()
                if event == "status":
                    self.status_var.set(str(value))
                elif event == "connection_error":
                    self._serial_exchange_verified = False
                    self.connection_var.set("Serial: disconnected")
                    self.status_var.set(str(value))
                elif event == "serial_exchange_verified" and self.serial.connected:
                    self._serial_exchange_verified = True
                    self.connection_var.set(
                        f"Serial: {self._connected_port} @ {self._connected_baudrate} baud; Arduino exchange verified"
                    )
                elif event == "camera_started":
                    self.status_var.set("Camera preview running; inference occurs only on request.")
                elif event == "camera_error":
                    self.running = False
                    self._camera_stop.set()
                    with self._frame_lock:
                        self.latest_rgb_frame = None
                    self.camera_label.configure(image="", text="Camera unavailable")
                    self._photo = None
                    self.status_var.set(str(value))
                    messagebox.showerror("Camera error", str(value))
                elif event == "prediction":
                    prediction, category = value
                    self._show_prediction(prediction, category)
                elif event == "manual_error":
                    messagebox.showerror("Inference error", str(value))
        except queue.Empty:
            pass
        if self.winfo_exists():
            self.after(50, self._drain_ui_events)

    def classify_current_frame(self) -> None:
        with self._frame_lock:
            frame = None if self.latest_rgb_frame is None else self.latest_rgb_frame.copy()
        if frame is None:
            messagebox.showwarning("Inference", "Start the camera and wait for a frame first.")
            return
        if self._manual_thread is not None and self._manual_thread.is_alive():
            self.status_var.set("A manual classification is already running.")
            return
        self._manual_thread = threading.Thread(
            target=self._classify_manual_frame,
            args=(frame,),
            name="manual-classifier",
            daemon=True,
        )
        self._manual_thread.start()
        self.status_var.set("Classifying current frame...")

    def _classify_manual_frame(self, frame) -> None:
        try:
            with self._inference_lock:
                prediction = self.predictor.predict_array(frame)
            category = map_prediction(prediction.predicted_class)
            try:
                self.audit_log.log(prediction, category, "manual_capture")
            except Exception as exc:
                self._ui_events.put(("status", f"Prediction log write failed: {exc}"))
            self._ui_events.put(("prediction", (prediction, category)))
            self._ui_events.put(("status", "Current frame classified on user request."))
        except Exception as exc:
            try:
                self.audit_log.log_failure(None, exc)
            except Exception as log_exc:
                self._ui_events.put(("status", f"Failure log write failed: {log_exc}"))
            self._ui_events.put(("manual_error", str(exc)))

    def _show_prediction(self, result, bin_index: int) -> None:
        self.count += 1
        self.prediction_var.set(result.predicted_class.title())
        self.confidence_var.set(f"{result.confidence:.1%}")
        self.time_var.set(f"{result.inference_time_ms:.1f} ms")
        bin_name = {0: "Yellow", 1: "Red", 2: "Blue", 3: "White"}.get(bin_index, "Unknown")
        self.category_var.set(f"Category {bin_index} - {bin_name} bin")
        self.counter_var.set(str(self.count))

    def stop_camera(self) -> None:
        self.running = False
        self._camera_stop.set()
        with self._frame_lock:
            self.latest_rgb_frame = None
        self.camera_label.configure(image="", text="Camera stopped")
        self._photo = None
        self.status_var.set("Camera stopped.")

    def connect_hardware(self) -> None:
        if self.serial.connected:
            self.status_var.set("Serial port is already open.")
            return
        if any(worker is not None and worker.is_alive() for worker in (
            self._listener_thread, self._processor_thread
        )):
            self.status_var.set("Serial workers are still stopping; try connecting again shortly.")
            return

        port = self.port_var.get().strip()
        if not port:
            ports = SerialController.available_ports()
            port = ports[0]["device"] if ports else ""
        if not port:
            messagebox.showwarning("Hardware", "Enter a COM port or connect a supported device first.")
            return
        try:
            baudrate = int(self.baud_var.get().strip())
        except ValueError:
            messagebox.showerror("Serial settings", "Baud rate must be a positive integer.")
            return
        if not 300 <= baudrate <= 2_000_000:
            messagebox.showerror("Serial settings", "Baud rate must be between 300 and 2,000,000.")
            return

        try:
            self.serial.connect(SerialSettings(port=port, baudrate=baudrate, timeout_seconds=0.1))
            self._connected_port = port
            self._connected_baudrate = baudrate
            self._serial_exchange_verified = False
            self.connection_var.set(
                f"Serial port open ({port} @ {baudrate} baud); Arduino exchange unverified"
            )
            self._start_serial_workers()
        except (ConnectionError, ValueError) as exc:
            self.connection_var.set("Serial: disconnected")
            messagebox.showerror("Hardware connection", str(exc))

    def _stop_serial_workers(self) -> None:
        self._listener_stop.set()
        self.serial.disconnect()
        self._request_queue.put(None)

    def disconnect_hardware(self) -> None:
        self._stop_serial_workers()
        self._serial_exchange_verified = False
        self.connection_var.set("Serial: disconnected")

    def export_logs(self) -> None:
        destination = Path.cwd() / "prediction_logs_export.csv"
        if not DEFAULT_LOG_PATH.exists():
            messagebox.showinfo("Export logs", "No prediction logs exist yet.")
            return
        shutil.copy2(DEFAULT_LOG_PATH, destination)
        self.status_var.set(f"Logs exported to {destination.name}")

    def clear_logs(self) -> None:
        if messagebox.askyesno("Clear logs", "Delete all recorded prediction events?"):
            self.audit_log.clear()
            self.status_var.set("Prediction logs cleared.")

    def close(self) -> None:
        self.stop_camera()
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=1.0)
            if self._camera_thread.is_alive() and self.capture is not None:
                self.capture.release()
        self._stop_serial_workers()
        self.destroy()

def main() -> None:
    """Start the desktop dashboard."""
    BiomedicalWasteDashboard().mainloop()


if __name__ == "__main__":
    main()
