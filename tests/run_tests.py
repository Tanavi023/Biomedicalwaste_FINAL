"""Run automated tests and record a concise Phase 16 report."""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = PROJECT_ROOT / "results" / "testing" / "test_report.json"


def main() -> None:
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"), pattern="test_*.py", top_level_dir=str(PROJECT_ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "tests_run": result.testsRun,
              "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
              "status": "passed" if result.wasSuccessful() else "failed"}
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Test report written to: {REPORT_PATH}")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
