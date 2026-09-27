"""Tkinter dashboard with preview-only camera frames and request-driven inference."""
from __future__ import annotations
import json
import shutil
import tkinter as tk
import uuid
from pathlib import Path
from tkinter import messagebox, ttk
import cv2
from PIL import Image, ImageTk
from src.communication.json_protocol import ProtocolError, error_response, parse_request_line, prediction_response
from src.communication.serial_controller import SerialController, SerialSettings
from src.inference.predict_image import DEFAULT_MODEL_PATH, BiomedicalWastePredictor
from src.utils.prediction_logger import DEFAULT_LOG_PATH, PredictionLogger

class BiomedicalWasteDashboard(tk.Tk):
    """Supervised UI; camera preview never predicts or actuates."""

    def __init__(self) -> None:
        super().__init__()
        self.title("AI-Driven Biomedical Waste Segregator")
        self.geometry("1080x720")
        self.minsize(900, 620)
        self.configure(bg="#F4F7FB")
        self.predictor = BiomedicalWastePredictor(DEFAULT_MODEL_PATH)
        self.serial = SerialController()
        self.audit_log = PredictionLogger()
        self.capture: cv2.VideoCapture | None = None
        self.running = False
        self.count = 0
        self.latest_rgb_frame = None
        self._photo: ImageTk.PhotoImage | None = None
        self._recent_object_ids: set[str] = set()
        self._recent_object_order: list[str] = []
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), background="#F4F7FB", foreground="#0B2D4D")
        style.configure("Metric.TLabel", font=("Segoe UI", 14, "bold"), background="white", foreground="#0B2D4D")
        header = ttk.Frame(self, padding=(24, 18))
        header.pack(fill="x")
        ttk.Label(header, text="AI-Driven Biomedical Waste Segregator", style="Title.TLabel").pack(side="left")
        self.connection_var = tk.StringVar(value="Hardware: disconnected")
        ttk.Label(header, textvariable=self.connection_var).pack(side="right")
        body = ttk.Frame(self, padding=(24, 0, 24, 20))
        body.pack(fill="both", expand=True)
        camera_frame = ttk.LabelFrame(body, text="Live Camera Preview (no automatic inference)", padding=10)
        camera_frame.pack(side="left", fill="both", expand=True, padx=(0, 12))
        self.camera_label = ttk.Label(camera_frame, text="Camera stopped", anchor="center")
        self.camera_label.pack(fill="both", expand=True)
        panel = ttk.Frame(body, width=330)
        panel.pack(side="right", fill="y")
        self.prediction_var = tk.StringVar(value=" ")
        self.confidence_var = tk.StringVar(value=" ")
        self.time_var = tk.StringVar(value=" ")
        self.category_var = tk.StringVar(value=" ")
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
        ttk.Entry(controls, textvariable=self.port_var).pack(fill="x", pady=(10, 3))
        ttk.Button(controls, text="Connect Hardware", command=self.connect_hardware).pack(fill="x", pady=3)
        ttk.Button(controls, text="Disconnect Hardware", command=self.disconnect_hardware).pack(fill="x", pady=3)
        ttk.Button(controls, text="Export Logs", command=self.export_logs).pack(fill="x", pady=(10, 3))
        ttk.Button(controls, text="Clear Logs", command=self.clear_logs).pack(fill="x", pady=3)
        self.status_var = tk.StringVar(value="System ready. Model loaded.")
        ttk.Label(self, textvariable=self.status_var, padding=(24, 8), anchor="w").pack(fill="x")

    def start_camera(self) -> None:
        if self.running:
            return
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            messagebox.showerror("Camera error", "Could not open the default webcam.")
            return
        self.running = True
        self.status_var.set("Camera preview running; inference occurs only on request.")
        self._update_frame()

    def _update_frame(self) -> None:
        if not self.running or self.capture is None:
            return
        ok, frame = self.capture.read()
        if not ok:
            self.status_var.set("Camera frame could not be read.")
            self.stop_camera()
            return
        self.latest_rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(self.latest_rgb_frame).resize((640, 480))
        self._photo = ImageTk.PhotoImage(image)
        self.camera_label.configure(image=self._photo, text="")
        self._poll_serial_request()
        self.after(30, self._update_frame)

    def _remember_object_id(self, object_id: str) -> bool:
        if object_id in self._recent_object_ids:
            return False
        self._recent_object_ids.add(object_id)
        self._recent_object_order.append(object_id)
        if len(self._recent_object_order) > 512:
            oldest = self._recent_object_order.pop(0)
            self._recent_object_ids.discard(oldest)
        return True

    def _poll_serial_request(self) -> None:
        if not self.serial.connected:
            return
        try:
            line = self.serial.read_request_line()
        except (ValueError, ConnectionError) as exc:
            self.status_var.set(f"Hardware request rejected: {exc}")
            try:
                self.serial.send_json_response(error_response(None, str(exc)))
            except ConnectionError:
                self.connection_var.set("Hardware: connection failed")
            return
        if line is None:
            return
        object_id = None
        try:
            object_id = parse_request_line(line)
            if not self._remember_object_id(object_id):
                response = error_response(object_id, "Duplicate object_id ignored.")
            elif self.latest_rgb_frame is None:
                response = error_response(object_id, "Camera frame is not available.")
            else:
                prediction = self.predictor.predict_array(self.latest_rgb_frame)
                response = prediction_response(object_id, prediction)
                self._show_prediction(prediction, response["bin"])
                self.audit_log.log(prediction, int(response["bin"]), "json_request_classified")
            self.serial.send_json_response(response)
            self.status_var.set(f"Processed hardware request {object_id}.")
        except (ProtocolError, ValueError) as exc:
            self.status_var.set(f"Hardware request rejected: {exc}")
            try:
                self.serial.send_json_response(error_response(object_id, str(exc)))
            except ConnectionError:
                self.connection_var.set("Hardware: connection failed")
        except ConnectionError as exc:
            self.status_var.set(str(exc))
            self.connection_var.set("Hardware: connection failed")

    def classify_current_frame(self) -> None:
        if self.latest_rgb_frame is None:
            messagebox.showwarning("Inference", "Start the camera and wait for a frame first.")
            return
        try:
            prediction = self.predictor.predict_array(self.latest_rgb_frame)
            from src.decision.mapper import map_prediction
            self._show_prediction(prediction, map_prediction(prediction.predicted_class))
            self.audit_log.log(prediction, map_prediction(prediction.predicted_class), "manual_capture")
            self.status_var.set("Current frame classified on user request.")
        except (ValueError, OSError) as exc:
            messagebox.showerror("Inference error", str(exc))

    def _show_prediction(self, result, bin_index: int) -> None:
        self.count += 1
        self.prediction_var.set(result.predicted_class.title())
        self.confidence_var.set(f"{result.confidence:.1%}")
        self.time_var.set(f"{result.inference_time_ms:.1f} ms")
        self.category_var.set(f"Bin {bin_index}")
        self.counter_var.set(str(self.count))

    def stop_camera(self) -> None:
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.latest_rgb_frame = None
        self.camera_label.configure(image="", text="Camera stopped")
        self._photo = None
        self.status_var.set("Camera stopped.")

    def connect_hardware(self) -> None:
        port = self.port_var.get().strip()
        if not port:
            ports = SerialController.available_ports()
            port = ports[0]["device"] if ports else ""
        if not port:
            messagebox.showwarning("Hardware", "Enter a COM port or connect a supported device first.")
            return
        try:
            self.serial.connect(SerialSettings(port=port))
            self.connection_var.set(f"Hardware: connected ({port})")
        except ConnectionError as exc:
            messagebox.showerror("Hardware connection", str(exc))

    def disconnect_hardware(self) -> None:
        self.serial.disconnect()
        self.connection_var.set("Hardware: disconnected")

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
        self.serial.disconnect()
        self.destroy()

def main() -> None:
    """Start the desktop dashboard."""
    BiomedicalWasteDashboard().mainloop()

if __name__ == "__main__":
    main()
