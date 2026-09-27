"""Verify split folders and manifest integrity after dataset splitting."""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.preprocessing.inspect_dataset import load_expected_classes
from src.utils.logging_config import configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS_FILE = PROJECT_ROOT / "data" / "labels.json"
SPLIT_DIRECTORIES = {
    "train": PROJECT_ROOT / "data" / "train",
    "validation": PROJECT_ROOT / "data" / "validation",
    "test": PROJECT_ROOT / "data" / "test",
}
MANIFEST_PATH = PROJECT_ROOT / "results" / "dataset_splits" / "split_manifest.csv"
VERIFICATION_PATH = PROJECT_ROOT / "results" / "dataset_splits" / "split_verification.json"
LOG_FILE = PROJECT_ROOT / "logs" / "dataset_split.log"
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def verify_split() -> int:
    """Check file existence, class coverage, counts, and content-hash leakage."""
    logger = configure_logging(LOG_FILE).getChild("split_verification")
    if not MANIFEST_PATH.is_file():
        logger.error("Split manifest is missing: %s", MANIFEST_PATH)
        return 2

    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    expected_classes = load_expected_classes(LABELS_FILE)
    raw_source_paths = {
        str(path.relative_to(PROJECT_ROOT))
        for class_name in expected_classes
        for path in (PROJECT_ROOT / "data" / "raw" / class_name).iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    }
    manifest_source_paths = {row["source_path"] for row in rows}
    missing_destinations = [row["destination_path"] for row in rows if not (PROJECT_ROOT / row["destination_path"]).is_file()]
    missing_sources = [row["source_path"] for row in rows if not (PROJECT_ROOT / row["source_path"]).is_file()]
    unassigned_source_paths = sorted(raw_source_paths - manifest_source_paths)
    duplicate_source_assignments = len(rows) - len(manifest_source_paths)
    counts = Counter((row["split"], row["class_name"]) for row in rows)
    hashes_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        hashes_to_splits[row["sha256"]].add(row["split"])
    leaked_hashes = {checksum: sorted(splits) for checksum, splits in hashes_to_splits.items() if len(splits) > 1}
    empty_class_folders = [
        f"{split_name}/{class_name}"
        for split_name, split_dir in SPLIT_DIRECTORIES.items()
        for class_name in expected_classes
        if counts[(split_name, class_name)] == 0 or not (split_dir / class_name).is_dir()
    ]
    report = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_row_count": len(rows),
        "split_class_counts": {
            split_name: {class_name: counts[(split_name, class_name)] for class_name in expected_classes}
            for split_name in SPLIT_DIRECTORIES
        },
        "missing_destination_count": len(missing_destinations),
        "missing_source_count": len(missing_sources),
        "unassigned_source_count": len(unassigned_source_paths),
        "duplicate_source_assignment_count": duplicate_source_assignments,
        "cross_split_duplicate_hash_count": len(leaked_hashes),
        "empty_or_missing_class_folders": empty_class_folders,
        "is_valid": (
            not missing_destinations
            and not missing_sources
            and not unassigned_source_paths
            and not duplicate_source_assignments
            and not leaked_hashes
            and not empty_class_folders
        ),
    }
    VERIFICATION_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["is_valid"]:
        logger.info("Split verification passed for %s files", len(rows))
        return 0
    logger.error("Split verification failed: %s", report)
    return 1


if __name__ == "__main__":
    raise SystemExit(verify_split())
