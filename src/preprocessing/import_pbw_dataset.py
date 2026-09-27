"""Import the Pharmaceutical and Biomedical Waste dataset into project classes.

The source dataset is organized by item type.  This module applies a documented
item-to-disposal-colour mapping and copies only defensibly labelled images into
``data/raw``.  It never deletes source images or existing project data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import shutil
import tempfile
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.utils.logging_config import configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE = PROJECT_ROOT / "data" / "pbw-source.zip"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "results" / "dataset_imports"
LOG_FILE = PROJECT_ROOT / "logs" / "dataset_import.log"
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

SOURCE_NAME = "Pharmaceutical and Biomedical Waste (PBW)"
SOURCE_URL = "https://www.kaggle.com/datasets/engineeringubu/pharmaceutical-and-biomedical-waste"
SOURCE_LICENSE = "CC BY-NC-SA 4.0"

# These mappings follow the intended Indian biomedical-waste disposal stream.
# Ambiguous non-biomedical categories are deliberately excluded, not guessed.
SOURCE_TO_TARGET_CLASS = {
    "(BT) Body Tissue or Organ": "yellow",
    "Gauze": "yellow",
    "Mask": "yellow",
    "Gloves": "red",
    "(PE) Plastic equipment-packaging": "red",
    "Syringe": "red",
    "(GE) Glass equipment-packaging 551": "blue",
    "(ME) Metal equipment -packaging": "blue",
    "(SN) Syringe needles": "white",
    "Tweezers": "white",
}
EXCLUDED_SOURCE_CLASSES = {
    "(OW) Organic wastes": "General organic material is not reliably biomedical waste.",
    "(PP) Paper equipment-packaging": "Paper packaging has no defensible target colour among the four requested classes.",
}


def sha256_file(path: Path) -> str:
    """Calculate a SHA-256 digest without loading an archive into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalise_name(value: str) -> str:
    """Create a stable ASCII filename component from a source class name."""
    normalised = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", normalised.lower()).strip("_")


def source_class_from_entry(entry: zipfile.ZipInfo) -> str | None:
    """Return the item class from the PBW archive's known folder layout."""
    parts = Path(entry.filename).parts
    if len(parts) < 4 or entry.is_dir():
        return None
    return parts[1]


def image_entries(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Return only supported image files from the source archive."""
    return [
        entry
        for entry in archive.infolist()
        if not entry.is_dir() and Path(entry.filename).suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def assert_safe_target(allow_existing: bool) -> None:
    """Prevent an accidental merge with a raw dataset from another source."""
    existing_files = [
        path
        for class_name in sorted(set(SOURCE_TO_TARGET_CLASS.values()))
        for path in (RAW_DATA_DIR / class_name).rglob("*")
        if path.is_file()
    ]
    if existing_files and not allow_existing:
        raise RuntimeError(
            "data/raw already contains images. Review their provenance before merging datasets, "
            "or rerun only when an intentional merge is required."
        )


def copy_entry(
    archive: zipfile.ZipFile,
    entry: zipfile.ZipInfo,
    target_class: str,
) -> tuple[Path, str, str]:
    """Copy one image by content hash, returning its path, hash, and action."""
    target_dir = RAW_DATA_DIR / target_class
    target_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(entry.filename).suffix.lower()

    with tempfile.NamedTemporaryFile(dir=target_dir, suffix=".part", delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
        digest = hashlib.sha256()
        with archive.open(entry) as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
                temporary_file.write(chunk)

    checksum = digest.hexdigest()
    destination = target_dir / f"pbw_{normalise_name(source_class_from_entry(entry) or 'unknown')}_{checksum[:16]}{suffix}"
    if destination.exists():
        temporary_path.unlink(missing_ok=True)
        return destination, checksum, "duplicate_skipped"

    shutil.move(str(temporary_path), str(destination))
    return destination, checksum, "imported"


def raw_pbw_counts() -> Counter[str]:
    """Count PBW-imported files currently present in each raw class directory."""
    return Counter(
        {
            class_name: sum(1 for path in (RAW_DATA_DIR / class_name).glob("pbw_*") if path.is_file())
            for class_name in sorted(set(SOURCE_TO_TARGET_CLASS.values()))
        }
    )


def create_provenance(
    archive: Path,
    source_counts: Counter[str],
    imported_counts: Counter[str],
    current_raw_counts: Counter[str],
) -> dict[str, object]:
    """Build the dataset provenance record needed for reproducibility."""
    return {
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"name": SOURCE_NAME, "url": SOURCE_URL, "license": SOURCE_LICENSE},
        "archive": {"filename": archive.name, "sha256": sha256_file(archive)},
        "source_to_target_class": SOURCE_TO_TARGET_CLASS,
        "excluded_source_classes": EXCLUDED_SOURCE_CLASSES,
        "source_class_counts": dict(sorted(source_counts.items())),
        "new_files_imported_in_this_run": dict(sorted(imported_counts.items())),
        "current_raw_class_counts": dict(sorted(current_raw_counts.items())),
    }


def run_import(archive_path: Path, dry_run: bool, allow_existing: bool) -> int:
    """Validate an archive and import all approved source classes into data/raw."""
    logger = configure_logging(LOG_FILE).getChild("pbw_import")
    if not archive_path.is_file():
        logger.error("Dataset archive does not exist: %s", archive_path)
        return 2

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            invalid_member = archive.testzip()
            if invalid_member:
                logger.error("Archive integrity check failed at: %s", invalid_member)
                return 3
            entries = image_entries(archive)
            source_counts = Counter(filter(None, (source_class_from_entry(entry) for entry in entries)))
            unmapped = sorted(set(source_counts) - set(SOURCE_TO_TARGET_CLASS) - set(EXCLUDED_SOURCE_CLASSES))
            if unmapped:
                logger.error("Unmapped source classes require review: %s", ", ".join(unmapped))
                return 4

            planned_counts = Counter()
            for source_class, count in source_counts.items():
                target_class = SOURCE_TO_TARGET_CLASS.get(source_class)
                if target_class:
                    planned_counts[target_class] += count
            logger.info("Validated archive with %s image entries", len(entries))
            logger.info("Planned import counts: %s", dict(sorted(planned_counts.items())))
            if dry_run:
                return 0

            assert_safe_target(allow_existing)
            manifest_rows: list[dict[str, object]] = []
            imported_counts = Counter()
            for entry in entries:
                source_class = source_class_from_entry(entry)
                target_class = SOURCE_TO_TARGET_CLASS.get(source_class or "")
                if not target_class:
                    continue
                destination, checksum, action = copy_entry(archive, entry, target_class)
                if action == "imported":
                    imported_counts[target_class] += 1
                manifest_rows.append(
                    {
                        "source_archive_path": entry.filename,
                        "source_class": source_class,
                        "target_class": target_class,
                        "destination_path": str(destination.relative_to(PROJECT_ROOT)),
                        "sha256": checksum,
                        "action": action,
                    }
                )
    except (OSError, zipfile.BadZipFile) as error:
        logger.error("Cannot read dataset archive: %s", error)
        return 3

    manifest_path = REPORTS_DIR / "pbw_import_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest_rows[0]) if manifest_rows else [])
        if manifest_rows:
            writer.writeheader()
            writer.writerows(manifest_rows)
    provenance_path = REPORTS_DIR / "pbw_provenance.json"
    provenance_path.write_text(
        json.dumps(create_provenance(archive_path, source_counts, imported_counts, raw_pbw_counts()), indent=2),
        encoding="utf-8",
    )
    logger.info("Imported counts: %s", dict(sorted(imported_counts.items())))
    logger.info("Manifest written to %s", manifest_path)
    return 0


def parse_args() -> argparse.Namespace:
    """Parse dataset-import options."""
    parser = argparse.ArgumentParser(description="Import the PBW dataset into biomedical disposal classes.")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--dry-run", action="store_true", help="Validate and report planned counts without copying images.")
    parser.add_argument("--allow-existing", action="store_true", help="Allow an intentional merge with existing raw data.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(run_import(arguments.archive, arguments.dry_run, arguments.allow_existing))
