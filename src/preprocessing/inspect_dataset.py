"""Inspect and safely clean the raw biomedical waste image dataset.

The default mode is read-only: it writes reports but never changes raw images.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

from src.utils.logging_config import configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LABELS_FILE = PROJECT_ROOT / "data" / "labels.json"
REPORTS_DIR = PROJECT_ROOT / "results" / "dataset_inspection"
LOG_FILE = PROJECT_ROOT / "logs" / "dataset_inspection.log"
QUARANTINE_DIR = PROJECT_ROOT / "data" / "quarantine"
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class ImageRecord:
    """Verified metadata for one image in the raw dataset."""

    path: Path
    class_name: str
    file_size_bytes: int
    width: int
    height: int
    image_format: str
    sha256: str


def load_expected_classes(labels_file: Path) -> list[str]:
    """Read class names from labels.json while preserving numeric class order."""
    with labels_file.open("r", encoding="utf-8") as file:
        labels = json.load(file)
    return [labels[str(index)].lower() for index in sorted(map(int, labels))]


def hash_file(path: Path) -> str:
    """Return a SHA-256 checksum for exact duplicate detection."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_image(path: Path) -> tuple[int, int, str]:
    """Fully decode an image and return its dimensions and format."""
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        return image.width, image.height, image.format or "UNKNOWN"


def inspect_class_images(class_dir: Path, class_name: str) -> tuple[list[ImageRecord], list[dict[str, str]]]:
    """Inspect one class directory and separate valid images from invalid files."""
    valid_records: list[ImageRecord] = []
    invalid_files: list[dict[str, str]] = []
    for path in sorted(item for item in class_dir.rglob("*") if item.is_file()):
        relative_path = str(path.relative_to(PROJECT_ROOT))
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            invalid_files.append({"path": relative_path, "reason": "unsupported_extension"})
            continue
        try:
            width, height, image_format = verify_image(path)
            valid_records.append(ImageRecord(path, class_name, path.stat().st_size, width, height, image_format, hash_file(path)))
        except (UnidentifiedImageError, OSError, ValueError) as error:
            invalid_files.append({"path": relative_path, "reason": f"unreadable_image: {error}"})
    return valid_records, invalid_files


def find_duplicate_groups(records: Iterable[ImageRecord]) -> dict[str, list[ImageRecord]]:
    """Group byte-identical images using their SHA-256 checksums."""
    hashes: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        hashes[record.sha256].append(record)
    return {checksum: group for checksum, group in hashes.items() if len(group) > 1}


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    """Write a UTF-8 CSV with a stable schema, including empty reports."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def quarantine(paths: Iterable[Path], reason: str, logger: logging.Logger) -> None:
    """Move reviewed files out of raw data while retaining their class path."""
    moved = 0
    for path in paths:
        destination = QUARANTINE_DIR / reason / path.relative_to(RAW_DATA_DIR)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
        moved += 1
    logger.warning("Quarantined %s files for reason=%s", moved, reason)


def build_statistics(records: list[ImageRecord], classes: list[str]) -> dict[str, object]:
    """Calculate class distribution and image-property summaries."""
    counts = Counter(record.class_name for record in records)
    widths, heights = [r.width for r in records], [r.height for r in records]
    total, largest, smallest = len(records), max(counts.values(), default=0), min(counts.values(), default=0)
    ratio = round(largest / smallest, 2) if smallest else None
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_valid_images": total,
        "class_counts": {name: counts[name] for name in classes},
        "class_percentages": {name: round(counts[name] / total * 100, 2) if total else 0.0 for name in classes},
        "image_dimensions": {"min_width": min(widths, default=0), "max_width": max(widths, default=0), "min_height": min(heights, default=0), "max_height": max(heights, default=0)},
        "class_imbalance_ratio": ratio,
        "balancing_guidance": "Collect more real images for underrepresented classes; apply augmentation only after splitting." if ratio and ratio > 1.5 else "Class distribution is within the 1.5:1 review threshold.",
    }


def run_inspection(quarantine_invalid: bool, quarantine_duplicates: bool) -> int:
    """Run validation, duplicate detection, reports, and optional cleanup."""
    logger = configure_logging(LOG_FILE).getChild("dataset_inspection")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    classes = load_expected_classes(LABELS_FILE)
    missing = [name for name in classes if not (RAW_DATA_DIR / name).is_dir()]
    if missing:
        logger.error("Missing class directories: %s", ", ".join(missing))
        return 2
    records, invalid_files = [], []
    for class_name in classes:
        valid, invalid = inspect_class_images(RAW_DATA_DIR / class_name, class_name)
        records.extend(valid)
        invalid_files.extend(invalid)
    duplicate_groups = find_duplicate_groups(records)
    image_rows = [{"path": str(r.path.relative_to(PROJECT_ROOT)), "class_name": r.class_name, "width": r.width, "height": r.height, "format": r.image_format, "file_size_bytes": r.file_size_bytes, "sha256": r.sha256} for r in records]
    duplicate_rows = [{"sha256": checksum, "path": str(r.path.relative_to(PROJECT_ROOT)), "class_name": r.class_name, "recommended_action": "keep_first_or_review_cross_class_duplicate"} for checksum, group in duplicate_groups.items() for r in group]
    stats = build_statistics(records, classes)
    stats.update({"invalid_file_count": len(invalid_files), "duplicate_group_count": len(duplicate_groups)})
    write_csv(REPORTS_DIR / "valid_images.csv", image_rows, ["path", "class_name", "width", "height", "format", "file_size_bytes", "sha256"])
    write_csv(REPORTS_DIR / "invalid_files.csv", invalid_files, ["path", "reason"])
    write_csv(REPORTS_DIR / "duplicate_groups.csv", duplicate_rows, ["sha256", "path", "class_name", "recommended_action"])
    (REPORTS_DIR / "dataset_statistics.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if quarantine_invalid:
        quarantine((PROJECT_ROOT / row["path"] for row in invalid_files), "invalid", logger)
    if quarantine_duplicates:
        quarantine((r.path for group in duplicate_groups.values() for r in group[1:]), "duplicates", logger)
    logger.info("Inspection complete: valid=%s invalid=%s duplicate_groups=%s", len(records), len(invalid_files), len(duplicate_groups))
    if not records:
        logger.error("No valid images found. Add images to data/raw/<class_name> and rerun inspection.")
        return 1
    return 0


def parse_args() -> argparse.Namespace:
    """Parse explicit cleanup options."""
    parser = argparse.ArgumentParser(description="Inspect and safely clean raw image data.")
    parser.add_argument("--quarantine-invalid", action="store_true")
    parser.add_argument("--quarantine-duplicates", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run_inspection(args.quarantine_invalid, args.quarantine_duplicates))
