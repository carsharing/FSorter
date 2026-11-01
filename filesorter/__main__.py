"""Entry point.

    LocalFileSorter.exe                      open the window
    LocalFileSorter.exe --background         start hidden in the system tray and keep sorting new files
    LocalFileSorter.exe run --source A --dest B [--detail fine]     one sorting run without a window
    LocalFileSorter.exe watch --source A --dest B                   keep sorting without a window
    LocalFileSorter.exe selftest [--out result.json]                check models, OCR and offline mode
"""

from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path

from . import APP_NAME, __version__, offline

offline.enable()  # before anything else gets a chance to open a connection


def _setup_logging(verbose: bool = False) -> None:
    from .paths import log_dir

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh = logging.handlers.RotatingFileHandler(log_dir() / "filesorter.log", maxBytes=2_000_000, backupCount=3,
                                              encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)
    for noisy in ("PIL", "matplotlib", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    if sys.stderr is not None and verbose:
        root.addHandler(logging.StreamHandler())


def _settings_from_args(a):
    from .config import Settings

    s = Settings.load() if a.use_saved else Settings()
    if a.source:
        s.sources = [str(Path(p).expanduser()) for p in a.source]
    if a.dest:
        s.destination = str(Path(a.dest).expanduser())
    if a.detail:
        s.detail = a.detail
    if a.no_ocr:
        s.ocr = False
    if a.no_ai_naming:
        s.ai_naming = False
    if a.threads:
        s.threads = a.threads
    return s


def _console_progress():
    last = [0.0, ""]

    def cb(stage, done, total, msg=""):
        now = time.time()
        if stage != last[1] or now - last[0] > 1.0 or stage == "Done":
            last[0], last[1] = now, stage
            pct = f" {100 * done / total:5.1f}%" if total else ""
            line = f"[{stage}]{pct} {msg}"[:150]
            if sys.stdout is not None:
                print(line, flush=True)

    return cb


def cmd_run(a) -> int:
    from .engine import Engine, lower_priority

    s = _settings_from_args(a)
    lower_priority()
    eng = Engine(s, progress=_console_progress(), models=Path(a.models) if a.models else None)
    stats = eng.run()
    if sys.stdout is not None:
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    return 0


def cmd_watch(a) -> int:
    from .background import Watcher

    s = _settings_from_args(a)
    w = Watcher(s, models=Path(a.models) if a.models else None, progress=_console_progress())
    w.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        w.stop()
    return 0


def cmd_selftest(a) -> int:
    """Loads every model and runs it once. Exit code 0 = everything needed works."""
    import numpy as np
    from PIL import Image, ImageDraw

    from .models import ModelHub
    from .paths import models_dir

    result: dict = {"version": __version__, "offline_guard": offline.verify()}
    hub = ModelHub(Path(a.models) if a.models else models_dir(), threads=2, ocr=True, ai_naming=True)
    result["status"] = hub.status()
    try:
        v = hub.text.embed(["Facture fournisseur et paiement", "Supplier invoice and payment", "Team bowling"]) \
            if not hub.text.needs_fit else None
        if v is not None:
            result["text_similarity_fr_en"] = float(v[0] @ v[1])
            result["text_similarity_unrelated"] = float(v[0] @ v[2])
    except Exception as exc:  # noqa: BLE001
        result["text_error"] = repr(exc)
    img = Image.new("RGB", (640, 360), "white")
    dr = ImageDraw.Draw(img)
    dr.rectangle([20, 20, 620, 60], fill=(40, 90, 200))
    dr.text((30, 100), "Quarterly sales report 2024", fill="black")
    try:
        e = hub.visual.embed_images([img])
        result["image_embedding_dim"] = int(e.shape[1])
        L = hub.visual.label_embeddings()
        if L is not None:
            from .labels import LABELS

            result["image_top_label"] = LABELS[int(np.argmax(L @ e[0]))][1]
    except Exception as exc:  # noqa: BLE001
        result["image_error"] = repr(exc)
    try:
        ocr = hub.ocr
        if ocr is not None:
            big = img.resize((1280, 720))
            result["ocr_text"] = ocr.read(big)
    except Exception as exc:  # noqa: BLE001
        result["ocr_error"] = repr(exc)
    if a.llm:
        namer = hub.namer
        if namer is not None:
            result["llm_name"] = namer.name_topic(["facture", "fournisseur", "paiement"], ["Facture 2024-03 Acme"])
    ok = result["offline_guard"] and "text_error" not in result and "image_error" not in result
    if a.require_models:
        ok = ok and result["status"]["text_model_ok"] and result["status"]["image_model_ok"]
    result["ok"] = bool(ok)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    if sys.stdout is not None:
        print(text)
    return 0 if ok else 1


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="LocalFileSorter", description=f"{APP_NAME} – sorts files by content, offline")
    p.add_argument("--version", action="version", version=f"{APP_NAME} {__version__}")
    p.add_argument("--background", action="store_true", help="start in the system tray")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd")
    for name in ("run", "watch"):
        sp = sub.add_parser(name)
        sp.add_argument("--source", action="append", help="folder to sort (repeatable)")
        sp.add_argument("--dest", help="destination folder")
        sp.add_argument("--detail", choices=["coarse", "normal", "fine"])
        sp.add_argument("--no-ocr", action="store_true")
        sp.add_argument("--no-ai-naming", action="store_true")
        sp.add_argument("--threads", type=int, default=0)
        sp.add_argument("--models", help="models folder (default: next to the app)")
        sp.add_argument("--use-saved", action="store_true", help="start from the settings saved by the window")
    st = sub.add_parser("selftest")
    st.add_argument("--out")
    st.add_argument("--models")
    st.add_argument("--llm", action="store_true", help="also test the naming model")
    st.add_argument("--require-models", action="store_true")
    a = p.parse_args(argv)
    _setup_logging(a.verbose)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "watch":
        return cmd_watch(a)
    if a.cmd == "selftest":
        return cmd_selftest(a)
    from .gui import run_gui

    return run_gui(background=a.background)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    sys.exit(main())
