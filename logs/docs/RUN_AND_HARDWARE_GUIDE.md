# Run and Hardware Guide

This guide is the practical reference for running the AI-Driven Biomedical Waste Segregator after the project has been set up.

## 1. Before you start

Confirm that these files exist:

```text
models/biomedical_waste_mobilenetv2.keras
data/labels.json
```

The label-to-command mapping is fixed:

| Waste category | Model index | Hardware command |
| --- | ---: | ---: |
| Yellow Waste | 0 | `0` |
| Red Waste | 1 | `1` |
| Blue Waste | 2 | `2` |
| White Waste | 3 | `3` |

## 2. One-time installation

Open PowerShell inside the project folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell does not allow activation, run this once in the same window and then retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 3. Start the desktop application

```powershell
python -m src.app
```

Dashboard workflow:

1. Click **Start Camera**.
2. Place one biomedical-waste item clearly in the camera view.
3. Read the predicted category, confidence, inference time, and mapped bin command.
4. Click **Stop Camera** when finished.
5. Use **Export Logs** to create `prediction_logs_export.csv` in the project folder.

The application records every prediction in `logs/prediction_logs.csv`.

## 4. Alternative command-line workflows

Predict one image:

```powershell
python -m src.inference.predict_image path\to\image.jpg
```

Predict every supported image in a folder:

```powershell
python -m src.inference.predict_folder path\to\folder --recursive
```

This creates `prediction_results.csv` inside the input folder.

Run the real-time webcam window:

```powershell
python -m src.inference.webcam --camera 0
```

Webcam keyboard controls:

| Key | Action |
| --- | --- |
| `Q` or `Esc` | Exit safely |
| `H` | Show or hide prediction history |

Evaluate the trained model:

```powershell
python -m src.evaluation.evaluate_model
```

Run software checks before a demonstration:

```powershell
python -m tests.run_tests
python -m src.verification.final_check
```

## 5. Connect the hardware

The software communicates with a microcontroller through USB serial communication. Connect the microcontroller to the computer using a USB data cable. The board creates a Windows COM port such as `COM3` or `COM5`.

### Find the COM port

1. Connect the microcontroller by USB.
2. Open **Device Manager** in Windows.
3. Expand **Ports (COM & LPT)**.
4. Note the port name, for example `COM3`.

You can also use the detected-port list from Python:

```powershell
python -c "from src.communication.serial_controller import SerialController; print(SerialController.available_ports())"
```

### Connect from the dashboard

1. Start the dashboard with python -m src.app.
2. Enter the Arduino COM port, for example COM3, in the hardware field.
3. Click Connect Hardware and confirm the dashboard reports the port and 9600 baud.
4. Click Start Camera and wait for a live frame before triggering a capture.
5. The dashboard listens continuously in the background. Each valid capture request starts one inference and receives one newline-terminated JSON response.
6. Click Disconnect Hardware before unplugging the board.

### Arduino JSON request and response

The dashboard baud-rate field defaults to 9600, based on the existing project configuration, and is editable. The Arduino firmware source is not in this repository; inspect its Serial.begin(...) value and enter that same baud rate in Python. Serial settings are 8 data bits, no parity, one stop bit, UTF-8, and newline-delimited JSON. The Arduino sends:

    {"command":"capture","object_id":1}

Python preserves object_id and replies with exactly one success or failure response, each terminated by a newline. The current model predicts yellow, red, blue, or white; it does not independently identify metal.
Example success response:

    {"status":"success","object_id":1,"category":2,"label":"blue","confidence":0.95}

Example failure response:

    {"status":"failed","object_id":1,"category":0,"label":"unknown","confidence":0.0}
 The current model outputs yellow, red, blue, and white. It does not predict metal; metal and glass remain routing aliases for category 2, and the response reports the actual model label.

### Fill-level and placement-angle support

The repository contains no fill-level sensor input or telemetry protocol, so no percentage is shown or logged. It also contains no approved bin placement angles. Supply the actual sensor type, wiring, Arduino JSON fields, and approved angle values before adding either feature.

### Physical Arduino verification

1. Connect the Arduino Uno with a USB data cable. In Device Manager, note its COM port. Close Arduino Serial Monitor and any other program using the port.
2. Confirm the existing firmware starts serial at 9600 baud, reads newline-terminated JSON requests, and can send the bytes {"command":"capture","object_id":1} followed by a newline. Do not replace or modify the firmware for this Python-side check.
3. Connect a webcam, launch python -m src.app, start the camera, and ensure a frame is visible.
4. Enter the COM port and connect hardware. Verify the dashboard shows that port and 9600 baud.
5. Trigger the firmware request and inspect the response through the firmware's existing receive/debug path. Do not open another serial monitor while Python owns the COM port.
6. Verify exactly one JSON line arrives with status, the original object_id, category, label, and confidence, followed by a newline. For a failure-path check, stop the camera and trigger a new valid request; expect status failed, the same ID, category 0, label unknown, and confidence 0.0.
7. Disconnect in the dashboard before unplugging. Mocked automated tests do not count as physical hardware verification.
## 6. Troubleshooting

| Issue | Likely solution |
| --- | --- |
| `Could not open webcam` | Close other apps using the camera, then retry with `--camera 1` if an external camera is connected. |
| Hardware connection fails | Verify USB cable, COM port, board driver, and that no serial monitor is open. |
| No response to a valid request | Check that the camera has a frame and inference is available; confirm the Arduino uses newline framing and the same COM port and baud rate. |
| Wrong classification | Improve lighting, ensure one object is visible, and collect more representative training data before retraining. |
| Python module not found | Activate `.venv` and run `pip install -r requirements.txt` again. |

## 7. Safety checklist before a live demonstration

- Verify the tested class-to-bin mapping.
- Verify the request and response exchange with the empty mechanism before a demonstration.
- Keep an emergency stop or power disconnect available.
- Have a human operator confirm the predicted waste category before disposal.
- Never treat a model prediction as the sole safety decision.
