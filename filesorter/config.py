"""User settings, saved as JSON in the user's profile (no admin rights needed)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .paths import settings_path

DEFAULT_EXCLUDES = [
    "~$*", "*.tmp", "*.temp", "*.crdownload", "*.part", "*.partial", "*.lnk", "*.url", "Thumbs.db", "desktop.ini",
    ".DS_Store", "*.ini", "*.sys", "*.dll", "*.bak",
    ".git", ".svn", "node_modules", "__pycache__", ".venv", "venv", "$RECYCLE.BIN", "System Volume Information",
]

DETAIL_LEVELS = ("coarse", "normal", "fine")


@dataclass
class Settings:
    sources: list[str] = field(default_factory=list)
    destination: str = ""
    detail: str = "normal"              # how finely topics are split: coarse | normal | fine
    ocr: bool = True                    # read text in screenshots, scans and screen recordings
    ai_naming: bool = True              # use the local language model (if installed) to name folders
    background: bool = True             # keep sorting new files automatically
    autostart: bool = False             # start with Windows (per-user, no admin)
    rescan_minutes: int = 30            # safety-net full rescan interval in background mode
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    include_hidden: bool = False
    skip_cloud_only: bool = True        # don't read OneDrive "online-only" files (that would download them)
    max_file_mb: int = 0                # 0 = no limit
    threads: int = 0                    # 0 = automatic

    def worker_threads(self) -> int:
        if self.threads > 0:
            return self.threads
        return max(1, min(8, (os.cpu_count() or 2) - 1))

    def save(self, path: Path | None = None) -> None:
        path = path or settings_path()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or settings_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        s = cls(**{k: v for k, v in data.items() if k in known})
        if s.detail not in DETAIL_LEVELS:
            s.detail = "normal"
        return s
