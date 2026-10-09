# Testing guide

Run automated checks with `python -m tests.run_tests`. They cover decision mapping, CSV audit logging, JSONL capture handling, newline framing, and mocked serial disconnection without requiring physical hardware.

Manual acceptance checks:

| Area | Procedure | Expected result |
| --- | --- | --- |
| Single image | Run `predict_image` with a valid and an invalid image. | JSON prediction for valid input; descriptive error for invalid input. |
| Folder inference | Run `predict_folder` on a folder containing images. | `prediction_results.csv` records success or failure per file. |
| Webcam | Run `webcam` and test `H`, `Q`, and camera-unavailable behavior. | Overlay, history toggle, safe exit, and clear camera error. |
| Serial | Connect Arduino firmware that sends JSONL capture requests at 9600 baud. | One classification and one correlated newline-terminated JSON response per valid request. |
| Dashboard | Start/stop camera, connect/disconnect hardware, export and clear logs. | UI remains responsive and resources release on close. |

Run `python -m src.verification.final_check` before a demonstration. It verifies expected project files and imports without opening a webcam or serial device.
