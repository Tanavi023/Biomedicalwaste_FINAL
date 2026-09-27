# Testing guide

Run automated checks with `python -m tests.run_tests`. They cover decision mapping, CSV audit logging, and mocked serial acknowledgement behavior without requiring physical hardware.

Manual acceptance checks:

| Area | Procedure | Expected result |
| --- | --- | --- |
| Single image | Run `predict_image` with a valid and an invalid image. | JSON prediction for valid input; descriptive error for invalid input. |
| Folder inference | Run `predict_folder` on a folder containing images. | `prediction_results.csv` records success or failure per file. |
| Webcam | Run `webcam` and test `H`, `Q`, and camera-unavailable behavior. | Overlay, history toggle, safe exit, and clear camera error. |
| Serial | Connect a test microcontroller responding `ACK:<command>`. | Command and acknowledgement are logged; a disconnected device retries then reports failure. |
| Dashboard | Start/stop camera, connect/disconnect hardware, export and clear logs. | UI remains responsive and resources release on close. |

Run `python -m src.verification.final_check` before a demonstration. It verifies expected project files and imports without opening a webcam or serial device.
