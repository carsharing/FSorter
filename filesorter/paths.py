"""Where things live on disk."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import APP_ID

STATE_DIRNAME = "_FileSorter"          # inside the destination: database, caches, older reports
UNSORTED_DIRNAME = "_Unsorted"         # inside the destination: files the app could not place
DUPLICATE_DIRNAME = "duplicate"        # inside each category folder
REPORT_FILENAME = "Sorting report.html"
IS_WINDOWS = os.name == "nt"


def app_dir() -> Path:
    """Folder of the installed app (next to the .exe), or the project root when running from source."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def user_data_dir() -> Path:
    if IS_WINDOWS:
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    d = base / APP_ID
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    env = os.environ.get("FILESORTER_MODELS")
    if env:
        return Path(env)
    return app_dir() / "models"


def settings_path() -> Path:
    return user_data_dir() / "settings.json"


def log_dir() -> Path:
    d = user_data_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def long_path(p: str | os.PathLike) -> str:
    """On Windows, use the \\\\?\\ prefix for long paths so copies work beyond 260 characters."""
    s = os.fspath(p)
    if not IS_WINDOWS or len(s) < 240 or s.startswith("\\\\?\\"):
        return s
    s = os.path.abspath(s)
    if s.startswith("\\\\"):
        return "\\\\?\\UNC\\" + s[2:]
    return "\\\\?\\" + s


def is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.resolve().relative_to(other.resolve())
        return True
    except (ValueError, OSError):
        return False
