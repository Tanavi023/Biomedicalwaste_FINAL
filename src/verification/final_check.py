"""Run non-destructive production readiness checks and write a JSON report."""

from __future__ import annotations

import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = PROJECT_ROOT / "results" / "verification" / "final_build_report.json"
REQUIRED_PATHS = (
    "data/labels.json", "models/biomedical_waste_mobilenetv2.keras", "src/evaluation/evaluate_model.py",
    "src/inference/predict_image.py", "src/inference/predict_folder.py", "src/inference/webcam.py",
    "src/decision/mapper.py", "src/communication/serial_controller.py", "src/dashboard/dashboard.py",
    "src/optimization/optimize_model.py", "README.md", "requirements.txt", "LICENSE",
)
MODULES = ("src.decision.mapper", "src.communication.serial_controller", "src.inference.predict_image", "src.dashboard.dashboard")


def run_final_check() -> Path:
    """Check files and imports without opening devices, cameras, or the GUI."""
    missing = [path for path in REQUIRED_PATHS if not (PROJECT_ROOT / path).is_file()]
    import_errors: dict[str, str] = {}
    for module in MODULES:
        try:
            importlib.import_module(module)
        except Exception as error:  # Report every import failure without suppressing it silently.
            import_errors[module] = f"{type(error).__name__}: {error}"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(), "required_paths_missing": missing,
        "module_import_errors": import_errors, "status": "passed" if not missing and not import_errors else "failed",
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return REPORT_PATH


if __name__ == "__main__":
    output = run_final_check()
    print(f"Final verification report written to: {output}")
    report = json.loads(output.read_text(encoding="utf-8"))
    raise SystemExit(0 if report["status"] == "passed" else 1)
