"""Stable checkout paths shared by nested application modules and commands."""
from pathlib import Path
import os

PROJECT_ROOT = Path(os.environ.get("PPFCL_PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
CONFIGURATION_DIRECTORY = PROJECT_ROOT / "configuration"
DATASET_DIRECTORY = PROJECT_ROOT / "datasets" / "raw"
if not DATASET_DIRECTORY.is_dir():
    DATASET_DIRECTORY = PROJECT_ROOT / "project_archive/datasets/raw"
RUNTIME_STATE_DIRECTORY = PROJECT_ROOT / "state"
