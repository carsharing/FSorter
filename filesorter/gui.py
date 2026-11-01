"""Desktop window + system-tray icon (PySide6/Qt)."""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QColor, QDesktopServices, QFont, QIcon, QPalette, QPixmap
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QMainWindow, QMenu, QProgressBar, QPushButton,
                               QSizePolicy, QSystemTrayIcon, QVBoxLayout, QWidget)

from . import APP_ID, APP_NAME, __version__, autostart, icon, offline
from .background import Watcher
from .config import Settings
from .engine import estimate_copy_size
from .paths import REPORT_FILENAME, models_dir

log = logging.getLogger(__name__)
offline.enable()

DETAIL_CHOICES = [("Broad folders", "coarse"), ("Balanced (recommended)", "normal"), ("Very precise folders", "fine")]

STYLE = """
QMainWindow, QWidget#root { background: #f5f7fa; }
QLabel { color: #1d2330; }
QLabel#title { font-size: 20px; font-weight: 600; }
QLabel#subtitle { color: #5d6779; }
QLabel#muted { color: #5d6779; }
QLabel#offline { color: #1f8a5b; font-weight: 600; }
QFrame#card { background: #ffffff; border: 1px solid #e3e6ec; border-radius: 10px; }
QLabel#cardTitle { font-size: 14px; font-weight: 600; }
QListWidget, QLineEdit, QComboBox { border: 1px solid #d5dae3; border-radius: 6px; padding: 5px; background: #fff; }
QPushButton { border: 1px solid #cdd3dd; border-radius: 6px; padding: 6px 14px; background: #fff; }
QPushButton:hover { background: #eef2f8; }
QPushButton:disabled { color: #9aa3b2; }
QPushButton#primary { background: #2563eb; color: #fff; border: 1px solid #1f55c9; font-weight: 600; padding: 8px 22px; }
QPushButton#primary:hover { background: #1f55c9; }
QPushButton#primary:disabled { background: #9db7ee; border-color: #9db7ee; }
QProgressBar { border: 1px solid #d5dae3; border-radius: 6px; background: #fff; height: 14px; text-align: center; }
QProgressBar::chunk { background: #2563eb; border-radius: 5px; }
"""


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _open_path(path: Path) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


class Bridge(QObject):
    """Carries events from the sorting thread to the window (Qt signals are thread-safe)."""
    progress = Signal(str, object, object, str)   # object: byte counts exceed 32-bit ints
    finished = Signal(dict)
    error = Signal(str)
    estimate = Signal(object, object)


def _card(title: str) -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(8)
    t = QLabel(title)
    t.setObjectName("cardTitle")
    lay.addWidget(t)
    return f, lay


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings, app_icon: QIcon):
        super().__init__()
        self.s = settings
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon)
        self.resize(760, 820)
        self.bridge = Bridge()
        self.bridge.progress.connect(self._on_progress)
        self.bridge.finished.connect(self._on_finished)
        self.bridge.error.connect(self._on_error)
        self.bridge.estimate.connect(self._on_estimate)
        self.watcher = Watcher(settings, progress=self.bridge.progress.emit, on_finished=self.bridge.finished.emit,
                               on_error=self.bridge.error.emit)
        self.tray: QSystemTrayIcon | None = None
        self._told_about_tray = False
        self._quitting = False
        self._build()
        self._load_into_widgets()
        self._refresh_estimate()
        self._model_status()
        if self.s.background and self.s.sources and self.s.destination:
            self.watcher.start()
        self._tick = QTimer(self)
        self._tick.timeout.connect(self._refresh_buttons)
        self._tick.start(1000)

    # -- layout --------------------------------------------------------------------------------------------------
    def _build(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(12)

        head = QHBoxLayout()
        logo = QLabel()
        pm = QPixmap()
        pm.loadFromData(icon.png_bytes(96))
        logo.setPixmap(pm.scaled(44, 44, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        head.addWidget(logo)
        titles = QVBoxLayout()
        t = QLabel(APP_NAME)
        t.setObjectName("title")
        sub = QLabel("Copies your files into precise folders based on what is inside them.")
        sub.setObjectName("subtitle")
        titles.addWidget(t)
        titles.addWidget(sub)
        head.addLayout(titles, 1)
        self.offline_label = QLabel("● Offline – nothing leaves this computer")
        self.offline_label.setObjectName("offline")
        head.addWidget(self.offline_label, 0, Qt.AlignTop)
        outer.addLayout(head)

        # 1. sources
        c1, l1 = _card("1.  Folders to sort")
        self.sources = QListWidget()
        self.sources.setMinimumHeight(96)
        self.sources.setSelectionMode(QListWidget.ExtendedSelection)
        l1.addWidget(self.sources)
        row = QHBoxLayout()
        add = QPushButton("Add folder…")
        add.clicked.connect(self._add_source)
        rem = QPushButton("Remove")
        rem.clicked.connect(self._remove_source)
        row.addWidget(add)
        row.addWidget(rem)
        row.addStretch(1)
        note = QLabel("Originals are never moved or changed.")
        note.setObjectName("muted")
        row.addWidget(note)
        l1.addLayout(row)
        outer.addWidget(c1)

        # 2. destination
        c2, l2 = _card("2.  Where to put the sorted copies")
        row = QHBoxLayout()
        self.dest = QLineEdit()
        self.dest.setPlaceholderText(r"e.g. D:\Sorted files")
        self.dest.editingFinished.connect(self._dest_edited)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._pick_dest)
        row.addWidget(self.dest, 1)
        row.addWidget(browse)
        l2.addLayout(row)
        self.space = QLabel("")
        self.space.setObjectName("muted")
        l2.addWidget(self.space)
        outer.addWidget(c2)

        # 3. options
        c3, l3 = _card("3.  Options")
        grid = QGridLayout()
        grid.addWidget(QLabel("Folder detail"), 0, 0)
        self.detail = QComboBox()
        for label, _ in DETAIL_CHOICES:
            self.detail.addItem(label)
        self.detail.currentIndexChanged.connect(self._options_changed)
        grid.addWidget(self.detail, 0, 1)
        grid.setColumnStretch(2, 1)
        l3.addLayout(grid)
        self.ocr = QCheckBox("Read text in screenshots, scans and screen recordings (OCR)")
        self.ai = QCheckBox("Name folders with the local AI model")
        self.bg = QCheckBox("Keep sorting new files automatically in the background")
        self.auto = QCheckBox("Start automatically when I sign in to Windows")
        for cb in (self.ocr, self.ai, self.bg, self.auto):
            cb.toggled.connect(self._options_changed)
            l3.addWidget(cb)
        if os.name != "nt":
            self.auto.setEnabled(False)
            self.auto.setToolTip("Only available on Windows")
        self.models = QLabel("")
        self.models.setObjectName("muted")
        l3.addWidget(self.models)
        outer.addWidget(c3)

        # 4. run
        c4, l4 = _card("4.  Sort")
        row = QHBoxLayout()
        self.start = QPushButton("Start sorting")
        self.start.setObjectName("primary")
        self.start.clicked.connect(self._start)
        self.stop = QPushButton("Stop")
        self.stop.clicked.connect(self._stop)
        self.pause = QPushButton("Pause background sorting")
        self.pause.setCheckable(True)
        self.pause.toggled.connect(self._toggle_pause)
        row.addWidget(self.start)
        row.addWidget(self.stop)
        row.addStretch(1)
        row.addWidget(self.pause)
        l4.addLayout(row)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setRange(0, 1000)
        l4.addWidget(self.bar)
        self.stage = QLabel("Ready.")
        self.stage.setWordWrap(True)
        l4.addWidget(self.stage)
        self.detail_label = QLabel("")
        self.detail_label.setObjectName("muted")
        self.detail_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        l4.addWidget(self.detail_label)
        row = QHBoxLayout()
        self.open_report = QPushButton("Open report")
        self.open_report.clicked.connect(self._open_report)
        self.open_dest = QPushButton("Open sorted folder")
        self.open_dest.clicked.connect(self._open_dest)
        row.addWidget(self.open_report)
        row.addWidget(self.open_dest)
        row.addStretch(1)
        ver = QLabel(f"v{__version__}")
        ver.setObjectName("muted")
        row.addWidget(ver)
        l4.addLayout(row)
        outer.addWidget(c4)
        outer.addStretch(1)
        self.setCentralWidget(root)

    def _load_into_widgets(self) -> None:
        self.sources.clear()
        self.sources.addItems(self.s.sources)
        self.dest.setText(self.s.destination)
        idx = next((i for i, (_, v) in enumerate(DETAIL_CHOICES) if v == self.s.detail), 1)
        widgets = (self.detail, self.ocr, self.ai, self.bg, self.auto)
        for w in widgets:
            w.blockSignals(True)
        self.detail.setCurrentIndex(idx)
        self.ocr.setChecked(self.s.ocr)
        self.ai.setChecked(self.s.ai_naming)
        self.bg.setChecked(self.s.background)
        self.auto.setChecked(autostart.is_enabled() if os.name == "nt" else False)
        for w in widgets:
            w.blockSignals(False)

    # -- settings ------------------------------------------------------------------------------------------------
    def _save(self) -> None:
        self.s.sources = [self.sources.item(i).text() for i in range(self.sources.count())]
        self.s.destination = self.dest.text().strip()
        self.s.detail = DETAIL_CHOICES[self.detail.currentIndex()][1]
        self.s.ocr = self.ocr.isChecked()
        self.s.ai_naming = self.ai.isChecked()
        self.s.background = self.bg.isChecked()
        self.s.autostart = self.auto.isChecked()
        try:
            self.s.save()
        except OSError as exc:
            log.warning("could not save settings: %s", exc)
        self.watcher.update_settings(self.s)

    def _add_source(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose a folder to sort")
        if d:
            d = str(Path(d))
            existing = [self.sources.item(i).text() for i in range(self.sources.count())]
            if d not in existing:
                self.sources.addItem(d)
            self._save()
            self._refresh_estimate()

    def _remove_source(self) -> None:
        for it in self.sources.selectedItems():
            self.sources.takeItem(self.sources.row(it))
        self._save()
        self._refresh_estimate()

    def _pick_dest(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Choose where to put the sorted copies")
        if d:
            self.dest.setText(str(Path(d)))
            self._dest_edited()

    def _dest_edited(self) -> None:
        self._save()
        self._refresh_estimate()

    def _options_changed(self, *_):
        if self.auto.isEnabled():
            autostart.set_enabled(self.auto.isChecked())
        self._save()
        if self.s.background and self.s.sources and self.s.destination:
            self.watcher.start()

    def _refresh_estimate(self) -> None:
        if not self.s.sources or not self.s.destination:
            self.space.setText("")
            return
        self.space.setText("Measuring how much space the copies need…")
        snap = Settings(**{**self.s.__dict__})

        def work():
            total, free = estimate_copy_size(snap)
            self.bridge.estimate.emit(total, free)

        threading.Thread(target=work, daemon=True).start()

    def _on_estimate(self, total: int, free: int) -> None:
        msg = f"The copies need up to {_human(total)} · {_human(free)} free on the destination drive"
        if free and total > free:
            msg += "  —  not enough space: files will be copied until the drive is nearly full."
        self.space.setText(msg)

    def _model_status(self) -> None:
        md = models_dir()
        has_text = (md / "text").is_dir()
        has_clip = (md / "clip").is_dir()
        has_llm = any((md / "llm").glob("*.gguf")) if (md / "llm").is_dir() else False
        try:
            import rapidocr_onnxruntime  # noqa: F401

            has_ocr = True
        except Exception:  # noqa: BLE001
            has_ocr = False

        def mark(ok: bool, label: str, fallback: str) -> str:
            return f"{label} ✓" if ok else f"{label}: {fallback}"

        self.models.setText("Local models:  " + "  ·  ".join([
            mark(has_text, "Text", "built-in"), mark(has_clip, "Images", "basic"), mark(has_ocr, "OCR", "off"),
            mark(has_llm, "Folder names", "from keywords")]))
        self.models.setToolTip(f"Models folder: {md}")
        if not has_llm:
            self.ai.setEnabled(False)
            self.ai.setText("Name folders with the local AI model (not installed – names come from keywords)")
        if offline.is_enabled():
            self.offline_label.setText("● Offline – nothing leaves this computer")
        else:
            self.offline_label.setText("○ Offline guard not active")
            self.offline_label.setStyleSheet("color:#b76e00")

    # -- running -------------------------------------------------------------------------------------------------
    def _start(self) -> None:
        self._save()
        problems = []
        if not self.s.sources:
            problems.append("Add at least one folder to sort.")
        if not self.s.destination:
            problems.append("Choose where to put the sorted copies.")
        if problems:
            self.stage.setText(" ".join(problems))
            return
        self.stage.setText("Starting…")
        self.bar.setRange(0, 0)
        self.watcher.run_now()

    def _stop(self) -> None:
        self.watcher.cancel_current()
        self.stage.setText("Stopping after the current file…")

    def _toggle_pause(self, on: bool) -> None:
        if on:
            self.watcher.pause()
            self.pause.setText("Resume background sorting")
            self.stage.setText("Paused.")
        else:
            self.watcher.resume()
            self.pause.setText("Pause background sorting")

    def _refresh_buttons(self) -> None:
        running = self.watcher.running
        self.start.setEnabled(not running)
        self.stop.setEnabled(running)
        self.pause.setEnabled(self.s.background)
        report = Path(self.s.destination) / REPORT_FILENAME if self.s.destination else None
        self.open_report.setEnabled(bool(report and report.exists()))
        self.open_dest.setEnabled(bool(self.s.destination and Path(self.s.destination).exists()))
        if self.tray:
            self.tray.setToolTip(f"{APP_NAME} – {'sorting…' if running else 'paused' if self.watcher.paused.is_set() else 'watching for new files' if self.s.background else 'idle'}")

    def _on_progress(self, stage: str, done: int, total: int, msg: str) -> None:
        if total > 0:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int(1000 * min(1.0, done / total)))
            count = f" ({done:,} of {total:,})" if stage != "Copying files" else f" ({_human(done)} of {_human(total)})"
        else:
            if stage == "Done":
                self.bar.setRange(0, 1000)
            else:
                self.bar.setRange(0, 0)
            count = f" ({done:,} found)" if stage == "Scanning folders" and done else ""
        self.stage.setText(f"{stage}{count}")
        self.detail_label.setText(msg)

    def _on_finished(self, stats: dict) -> None:
        self.bar.setRange(0, 1000)
        self.bar.setValue(1000)
        if stats.get("cancelled"):
            self.stage.setText("Stopped. Already copied files stay in place; the next run continues from here.")
        else:
            copied = stats.get("files copied", 0)
            self.stage.setText(f"Done at {time.strftime('%H:%M')} – {copied:,} files copied, "
                               f"{stats.get('duplicates', 0):,} duplicates found.")
        self.detail_label.setText("")

    def _on_error(self, msg: str) -> None:
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)
        self.stage.setText(msg)

    def _open_report(self) -> None:
        _open_path(Path(self.s.destination) / REPORT_FILENAME)

    def _open_dest(self) -> None:
        _open_path(Path(self.s.destination))

    # -- tray ----------------------------------------------------------------------------------------------------
    def attach_tray(self, tray: QSystemTrayIcon) -> None:
        self.tray = tray
        menu = QMenu()
        for text, fn in (("Open " + APP_NAME, self.show_normal), ("Sort now", self._start),
                         ("Open report", self._open_report), ("Open sorted folder", self._open_dest)):
            a = QAction(text, menu)
            a.triggered.connect(fn)
            menu.addAction(a)
        self.tray_pause = QAction("Pause background sorting", menu)
        self.tray_pause.setCheckable(True)
        self.tray_pause.toggled.connect(self.pause.setChecked)
        self.pause.toggled.connect(self.tray_pause.setChecked)
        menu.addAction(self.tray_pause)
        menu.addSeparator()
        q = QAction("Quit", menu)
        q.triggered.connect(self.quit)
        menu.addAction(q)
        tray.setContextMenu(menu)
        tray.activated.connect(lambda reason: self.show_normal() if reason in (
            QSystemTrayIcon.Trigger, QSystemTrayIcon.DoubleClick) else None)
        self._tray_menu = menu

    def show_normal(self) -> None:
        self.show()
        self.setWindowState(self.windowState() & ~Qt.WindowMinimized)
        self.raise_()
        self.activateWindow()

    def closeEvent(self, event) -> None:
        if not self._quitting and self.tray is not None and (self.s.background or self.watcher.running):
            event.ignore()
            self.hide()
            if not self._told_about_tray:
                self._told_about_tray = True
                msg = ("Still sorting new files in the background." if self.s.background
                       else "Finishing the current run in the background.")
                self.tray.showMessage(APP_NAME, msg + " Use the tray icon to open or quit.",
                                      QSystemTrayIcon.Information, 4000)
            return
        self.quit()

    def quit(self) -> None:
        self._quitting = True
        self.watcher.stop()
        if self.tray:
            self.tray.hide()
        QApplication.quit()


def _apply_palette(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#f5f7fa"))
    pal.setColor(QPalette.Base, QColor("#ffffff"))
    pal.setColor(QPalette.Text, QColor("#1d2330"))
    pal.setColor(QPalette.WindowText, QColor("#1d2330"))
    pal.setColor(QPalette.Button, QColor("#ffffff"))
    pal.setColor(QPalette.ButtonText, QColor("#1d2330"))
    pal.setColor(QPalette.Highlight, QColor("#2563eb"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    app.setPalette(pal)
    app.setStyleSheet(STYLE)
    f = QFont("Segoe UI" if os.name == "nt" else app.font().family(), 10)
    app.setFont(f)


def run_gui(background: bool = False) -> int:
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:  # noqa: BLE001
            pass
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setQuitOnLastWindowClosed(False)

    # One instance only: a second launch just shows the running window (local named pipe, not network).
    server_name = f"{APP_ID}-{os.environ.get('USERNAME') or os.environ.get('USER') or 'user'}"
    sock = QLocalSocket()
    sock.connectToServer(server_name)
    if sock.waitForConnected(300):
        sock.write(b"show")
        sock.flush()
        sock.waitForBytesWritten(300)
        return 0
    QLocalServer.removeServer(server_name)
    server = QLocalServer()
    server.listen(server_name)

    _apply_palette(app)
    pm = QPixmap()
    pm.loadFromData(icon.png_bytes(256))
    app_icon = QIcon(pm)
    app.setWindowIcon(app_icon)
    settings = Settings.load()
    win = MainWindow(settings, app_icon)
    server.newConnection.connect(lambda: (server.nextPendingConnection(), win.show_normal()))

    if QSystemTrayIcon.isSystemTrayAvailable():
        tray = QSystemTrayIcon(app_icon)
        win.attach_tray(tray)
        tray.show()
    if not background or not (settings.sources and settings.destination):
        win.show()
    return app.exec()


def screenshot(path: str, settings: Settings | None = None) -> None:
    """Renders the window to an image (used for documentation)."""
    app = QApplication.instance() or QApplication(sys.argv)
    _apply_palette(app)
    pm = QPixmap()
    pm.loadFromData(icon.png_bytes(256))
    win = MainWindow(settings or Settings.load(), QIcon(pm))
    win.show()
    for _ in range(20):
        app.processEvents()
        time.sleep(0.05)
    win.grab().save(path)
    win.watcher.stop()


__all__ = ["run_gui", "screenshot"]
