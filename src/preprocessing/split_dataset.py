"""Create a deterministic, leakage-safe train/validation/test image split.

Raw images are never moved or modified.  This module copies them into split
folders and records the exact assignment in a CSV manifest for reproducibility.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.preprocessing.inspect_dataset import load_expected_classes
from src.utils.logging_config import configure_logging

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
LABELS_FILE = PROJECT_ROOT / "data" / "labels.json"
SPLIT_DIRECTORIES = {
    "train": PROJECT_ROOT / "data" / "train",
    "validation": PROJECT_ROOT / "data" / "validation",
    "test": PROJECT_ROOT / "data" / "test",
}
REPORTS_DIR = PROJECT_ROOT / "results" / "dataset_splits"
LOG_FILE = PROJECT_ROOT / "logs" / "dataset_split.log"
SUPPORTED_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


@dataclass(frozen=True)
class ImageGroup:
    """Files with identical bytes, which must be assigned to one split together."""

    checksum: str
    paths: tuple[Path, ...]


def sha256_file(path: Path) -> str:
    """Calculate a stable content fingerprint for leakage prevention."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_image_groups(class_dir: Path) -> list[ImageGroup]:
    """Group same-content images so duplicates cannot cross a dataset split."""
    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(class_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            grouped_paths[sha256_file(path)].append(path)
    return [ImageGroup(checksum, tuple(paths)) for checksum, paths in grouped_paths.items()]


def allocate_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    """Allocate integer image counts with the largest-remainder method."""
    raw_counts = {name: total * ratio for name, ratio in ratios.items()}
    counts = {name: math.floor(value) for name, value in raw_counts.items()}
    remaining = total - sum(counts.values())
    for name in sorted(ratios, key=lambda item: (raw_counts[item] - counts[item], ratios[item]), reverse=True)[:remaining]:
        counts[name] += 1
    return counts


def assign_groups(groups: list[ImageGroup], targets: dict[str, int], random_generator: random.Random) -> dict[str, list[ImageGroup]]:
    """Assign groups near target counts while keeping every group intact."""
    shuffled_groups = groups.copy()
    random_generator.shuffle(shuffled_groups)
    assignments: dict[str, list[ImageGroup]] = {name: [] for name in targets}
    actual_counts = Counter()
    for group in shuffled_groups:
        split_name = max(targets, key=lambda name: (targets[name] - actual_counts[name], -actual_counts[name]))
        assignments[split_name].append(group)
        actual_counts[split_name] += len(group.paths)
    return assignments


def target_split_has_files(classes: list[str]) -> bool:
    """Return whether any destination class folder already contains files."""
    return any(
        any((split_dir / class_name).glob("*"))
        for split_dir in SPLIT_DIRECTORIES.values()
        for class_name in classes
    )


def clear_existing_splits(classes: list[str]) -> None:
    """Remove only known split class folders when explicitly requested."""
    for split_dir in SPLIT_DIRECTORIES.values():
        for class_name in classes:
            target = split_dir / class_name
            if target.exists():
                shutil.rmtree(target)


def validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> dict[str, float]:
    """Validate and return a named split-ratio mapping."""
    ratios = {"train": train_ratio, "validation": validation_ratio, "test": test_ratio}
    if any(value <= 0 for value in ratios.values()) or not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Split ratios must be positive and sum to exactly 1.0.")
    return ratios


def write_reports(manifest_rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    """Write reproducible split-assignment and summary reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORTS_DIR / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["source_path", "class_name", "sha256", "split", "destination_path"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    (REPORTS_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def build_split(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
    overwrite: bool,
) -> int:
    """Create and report a deterministic stratified split from validated raw data."""
    logger = configure_logging(LOG_FILE).getChild("dataset_split")
    classes = load_expected_classes(LABELS_FILE)
    ratios = validate_ratios(train_ratio, validation_ratio, test_ratio)
    missing = [class_name for class_name in classes if not (RAW_DATA_DIR / class_name).is_dir()]
    if missing:
        logger.error("Missing raw class directories: %s", ", ".join(missing))
        return 2
    if target_split_has_files(classes):
        if not overwrite:
            logger.error("Split folders already contain files. Refusing to overwrite them without --overwrite.")
            return 3
        clear_existing_splits(classes)

    random_generator = random.Random(seed)
    manifest_rows: list[dict[str, object]] = []
    class_summary: dict[str, dict[str, int]] = {}
    for class_name in classes:
        groups = collect_image_groups(RAW_DATA_DIR / class_name)
        input_count = sum(len(group.paths) for group in groups)
        if input_count < len(ratios):
            logger.error("Class %s has too few images to populate every split.", class_name)
            return 4
        targets = allocate_counts(input_count, ratios)
        assignments = assign_groups(groups, targets, random_generator)
        actual_counts: dict[str, int] = {}
        for split_name, assigned_groups in assignments.items():
            destination_dir = SPLIT_DIRECTORIES[split_name] / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            actual_counts[split_name] = 0
            for group in assigned_groups:
                for source_path in group.paths:
                    destination_path = destination_dir / source_path.name
                    shutil.copy2(source_path, destination_path)
                    actual_counts[split_name] += 1
                    manifest_rows.append(
                        {
                            "source_path": str(source_path.relative_to(PROJECT_ROOT)),
                            "class_name": class_name,
                            "sha256": group.checksum,
                            "split": split_name,
                            "destination_path": str(destination_path.relative_to(PROJECT_ROOT)),
                        }
                    )
        class_summary[class_name] = {"source": input_count, **actual_counts}
        logger.info("Split %s: %s", class_name, class_summary[class_name])

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "ratios": ratios,
        "class_counts": class_summary,
        "totals": {
            split_name: sum(values[split_name] for values in class_summary.values())
            for split_name in SPLIT_DIRECTORIES
        },
    }
    write_reports(manifest_rows, summary)
    logger.info("Dataset split complete: %s", summary["totals"])
    return 0


def parse_args() -> argparse.Namespace:
    """Parse split settings and explicit overwrite consent."""
    parser = argparse.ArgumentParser(description="Create a deterministic train/validation/test dataset split.")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Delete only existing split class folders before rebuilding.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        build_split(
            train_ratio=args.train_ratio,
            validation_ratio=args.validation_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    )
