"""Local runtime settings that keep generated library state inside the project."""

from __future__ import annotations

import os
from pathlib import Path


def configure_runtime_environment() -> None:
    """Keep Matplotlib and Keras runtime caches within the project workspace."""
    project_root = Path(__file__).resolve().parents[2]
    cache_directory = project_root / "logs" / ".matplotlib"
    keras_directory = project_root / "models" / ".keras_cache"
    cache_directory.mkdir(parents=True, exist_ok=True)
    keras_directory.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(cache_directory)
    os.environ["KERAS_HOME"] = str(keras_directory)
