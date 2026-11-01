"""Background mode: watches the source folders and sorts new or changed files automatically, without prompts."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from .config import Settings
from .engine import Cancelled, Engine, lower_priority
from .paths import is_relative_to

log = logging.getLogger(__name__)

QUIET_SECONDS = 20        # wait until a folder has been quiet this long (downloads/copies finished)
_IGNORED_SUFFIXES = (".tmp", ".crdownload", ".part", ".partial", ".fs-partial", "~")


class Watcher:
    def __init__(self, settings: Settings, models: Path | None = None, progress=None, on_finished=None,
                 on_error=None):
        self.s = settings
        self.models = models
        self.progress = progress or (lambda *a: None)
        self.on_finished = on_finished or (lambda stats: None)
        self.on_error = on_error or (lambda msg: None)
        self._trigger = threading.Event()
        self._stop = threading.Event()
        self.paused = threading.Event()
        self._cancel = threading.Event()
        self._dirty_since: float | None = None
        self._last_run = time.time()
        self._thread: threading.Thread | None = None
        self._observer = None
        self.running = False
        self.last_stats: dict | None = None

    # -- control -------------------------------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="sorter", daemon=True)
        self._thread.start()
        self._start_observer()

    def stop(self) -> None:
        self._stop.set()
        self._cancel.set()
        self._trigger.set()
        self._stop_observer()
        if self._thread:
            self._thread.join(timeout=10)

    def run_now(self) -> None:
        self._cancel.clear()
        self._trigger.set()
        self.start()

    def cancel_current(self) -> None:
        self._cancel.set()

    def pause(self) -> None:
        self.paused.set()

    def resume(self) -> None:
        self.paused.clear()

    def update_settings(self, settings: Settings) -> None:
        self.s = settings
        self._stop_observer()
        self._start_observer()

    # -- file system events --------------------------------------------------------------------------------------
    def _start_observer(self) -> None:
        if not self.s.background or not self.s.sources or not self.s.destination:
            return
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            log.warning("watchdog not installed: relying on periodic rescans")
            return
        dest = Path(self.s.destination)
        watcher = self

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event):
                if event.is_directory and event.event_type == "modified":
                    return
                p = Path(getattr(event, "dest_path", "") or event.src_path)
                if p.name.lower().endswith(_IGNORED_SUFFIXES) or p.name.startswith("~$"):
                    return
                if is_relative_to(p, dest):
                    return
                watcher._dirty_since = time.time()

        obs = Observer()
        for src in self.s.sources:
            if Path(src).is_dir():
                try:
                    obs.schedule(Handler(), src, recursive=True)
                except OSError as exc:
                    log.warning("cannot watch %s: %s", src, exc)
        obs.daemon = True
        obs.start()
        self._observer = obs

    def _stop_observer(self) -> None:
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception:  # noqa: BLE001
                pass
            self._observer = None

    # -- loop ----------------------------------------------------------------------------------------------------
    def _due(self) -> bool:
        if self._trigger.is_set():
            return True
        if not self.s.background or self.paused.is_set():
            return False
        now = time.time()
        if self._dirty_since and now - self._dirty_since >= QUIET_SECONDS:
            return True
        return now - self._last_run >= max(5, self.s.rescan_minutes) * 60

    def _loop(self) -> None:
        lower_priority(background=True)
        while not self._stop.is_set():
            self._trigger.wait(timeout=5)
            if self._stop.is_set():
                break
            if not self._due():
                continue
            self._trigger.clear()
            self._dirty_since = None
            self._run_once()

    def _run_once(self) -> None:
        self.running = True
        self._cancel.clear()
        try:
            eng = Engine(self.s, progress=self.progress, cancel=self._cancel, pause=self.paused, models=self.models)
            stats = eng.run()
            self.last_stats = stats
            self.on_finished(stats)
        except Cancelled:
            pass
        except ValueError as exc:      # settings problem (missing folder...)
            self.on_error(str(exc))
        except Exception as exc:  # noqa: BLE001
            log.exception("sorting run failed")
            self.on_error(f"Sorting stopped because of an error: {exc}")
        finally:
            self._last_run = time.time()
            self.running = False
