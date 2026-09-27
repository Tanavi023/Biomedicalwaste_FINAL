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

1. Start the dashboard with `python -m src.app`.
2. Enter the COM port, for example `COM3`, in the hardware field.
3. Click **Connect Hardware**.
4. Confirm that the dashboard changes to `Hardware: connected (COM3)`.
5. Start the camera. Each valid classification is mapped to one command (`0`–`3`) and sent only while connected.
6. Click **Disconnect Hardware** before unplugging the board.

The default serial settings are:

| Setting | Value |
| --- | --- |
| Baud rate | `9600` |
| Data bits | `8` |
| Parity | None |
| Stop bits | `1` |
| Timeout | 1 second |
| Line ending | Newline (`\n`) |

### Microcontroller firmware contract

Your Arduino, ESP32, or compatible firmware must:

1. Start serial communication at **9600 baud**.
2. Read a newline-delimited command.
3. Accept only `0`, `1`, `2`, or `3`.
4. Move the correct servo, motor, or bin mechanism.
5. Send a non-empty acknowledgement line back to the computer, preferably `ACK:<command>`.

Expected serial conversation:

```text
Computer → microcontroller: 2\n
Microcontroller → computer: ACK:2\n
```

Example Arduino-compatible firmware skeleton:

```cpp
void setup() {
  Serial.begin(9600);
  // Initialize motors/servos here.
}

void loop() {
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "0" || command == "1" || command == "2" || command == "3") {
      // Move the bin mechanism for command here.
      Serial.println("ACK:" + command);
    } else {
      Serial.println("ERROR:INVALID_COMMAND");
    }
  }
}
```

Do not connect motors or servos directly to a microcontroller USB power pin. Use a suitable external power supply, shared ground, and the correct driver circuitry for the actuator. Test the mechanism with no biomedical waste present before live operation.

## 6. Troubleshooting

| Issue | Likely solution |
| --- | --- |
| `Could not open webcam` | Close other apps using the camera, then retry with `--camera 1` if an external camera is connected. |
| Hardware connection fails | Verify USB cable, COM port, board driver, and that no serial monitor is open. |
| Acknowledgement timeout | Check that the firmware calls `Serial.println(...)` after receiving each command. |
| Wrong classification | Improve lighting, ensure one object is visible, and collect more representative training data before retraining. |
| Python module not found | Activate `.venv` and run `pip install -r requirements.txt` again. |

## 7. Safety checklist before a live demonstration

- Verify the tested class-to-bin mapping.
- Test every command (`0`, `1`, `2`, `3`) manually with an empty mechanism.
- Keep an emergency stop or power disconnect available.
- Have a human operator confirm the predicted waste category before disposal.
- Never treat a model prediction as the sole safety decision.
