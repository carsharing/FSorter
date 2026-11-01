"""End-to-end: first run, then a later run with new / changed / duplicate files."""

import hashlib
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from filesorter.config import Settings
from filesorter.engine import Engine
from filesorter.paths import REPORT_FILENAME, UNSORTED_DIRNAME

ROOT = Path(__file__).resolve().parent.parent


def _digest(folder: Path) -> dict:
    return {str(p.relative_to(folder)): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def _copied(dest: Path) -> dict[str, Path]:
    return {p.name: p for p in dest.rglob("*") if p.is_file() and "_FileSorter" not in p.parts}


@pytest.fixture()
def workspace(sample, tmp_path):
    src = tmp_path / "src"
    shutil.copytree(sample, src)
    return src, tmp_path / "Sorted"


def _run(src, dest, models=None, **kw):
    s = Settings(sources=[str(src)], destination=str(dest), ocr=kw.pop("ocr", True), **kw)
    return Engine(s, models=models or (dest.parent / "no-models")).run()


def test_first_run_sorts_by_subject(workspace):
    src, dest = workspace
    before = _digest(src)
    stats = _run(src, dest)
    assert _digest(src) == before, "source files must never change"
    files = _copied(dest)
    # every non-empty source file was copied exactly once
    n_sources = sum(1 for p in src.rglob("*") if p.is_file() and p.stat().st_size > 0)
    assert len(files) - 1 == n_sources  # minus the report
    assert stats["files copied"] == n_sources

    def folder(name):
        return files[name].parent.relative_to(dest)

    # files about the same subject end up together, whatever their format or language
    same = [
        ["Note de frais juin.docx", "Remboursement mission Lyon.pdf", "Frais kilométriques mai.pptx",
         "RE Note de frais juin.eml"],
        ["Phishing awareness.docx", "Password policy.pdf", "RE Phishing awareness.eml"],
        ["Formation Excel module 1.docx", "Exercices tableaux croisés.pdf", "Quiz Excel.txt"],
        ["Contrat fournisseur Acme.docx", "Conditions générales achat.pdf"],
    ]
    for group in same:
        assert len({folder(n) for n in group}) == 1, group
    assert len({folder(g[0]) for g in same}) == len(same)
    # two levels deep
    assert len(folder("Quiz Excel.txt").parts) == 2

    # duplicates go to a "duplicate" folder inside the original's folder
    assert folder("brand guidelines - Copie.docx") == folder("Brand guidelines v2.docx") / "duplicate"
    assert files["photo equipe.jpg"].parent.name == "duplicate"
    assert files["photo equipe.jpg"].parent.parent == files["IMG_4101.JPG"].parent
    # recoloured logos are not duplicates
    assert all(files[f"logo_v{i}.png"].parent.name != "duplicate" for i in (1, 2, 3))

    # event photos grouped by date, named after the meaningful folder
    assert "2024-06-14" in files["IMG_4100.JPG"].parent.name
    assert "Annecy" in files["IMG_4100.JPG"].parent.name
    assert files["IMG-20241219-WA000.jpg"].parent != files["IMG_4100.JPG"].parent

    # things that can't be understood go to _Unsorted
    for n in ("setup_tool.exe", "data.bin", "broken.pdf"):
        assert UNSORTED_DIRNAME in files[n].parts, n

    # screenshots are read with OCR and sorted by subject
    assert folder("Screenshot 2024-05-02 101500.png") == folder("Budget prévisionnel 2025.docx")

    report = (dest / REPORT_FILENAME).read_text(encoding="utf-8")
    assert "Sorting report" in report and "Quiz Excel.txt" in report


def test_later_runs_are_incremental(workspace):
    src, dest = workspace
    _run(src, dest)
    first = _copied(dest)
    subprocess.run([sys.executable, str(ROOT / "tests" / "add_new_files.py"), str(src)], check=True)
    stats = _run(src, dest)
    files = _copied(dest)
    assert stats["new or changed files"] == 13
    # existing copies stay where they were
    for name, p in first.items():
        if name != REPORT_FILENAME:
            assert files[name] == p
    # a new expense note joins the expense folder
    assert files["Note de frais juillet.docx"].parent == files["Note de frais juin.docx"].parent
    # a new subject gets its own new folder
    csr = {files[n].parent for n in ("Sustainability report 2024.docx", "Carbon footprint assessment.docx",
                                     "ESG rating feedback.docx", "Climate targets roadmap.docx")}
    assert len(csr) == 1 and UNSORTED_DIRNAME not in next(iter(csr)).parts
    assert next(iter(csr)) not in {p.parent for p in first.values()}
    # photo on the same day as an existing event joins it; a new day makes a new event folder
    assert files["IMG_extra_annecy.jpg"].parent == files["IMG_4100.JPG"].parent
    assert "2025-03-07" in files["IMG_90.jpg"].parent.name
    # renamed duplicate of an existing file
    assert files["policy_final.pdf"].parent == files["Password policy.pdf"].parent / "duplicate"
    # a changed source file is re-copied in place
    assert "Question bonus" in files["Quiz Excel.txt"].read_text(encoding="utf-8")

    # nothing new: nothing is copied and no model needs loading
    stats = _run(src, dest)
    assert stats["files copied"] == 0 and stats["new or changed files"] == 0


def test_run_with_onnx_models(workspace, tmp_path):
    """Same pipeline through the neural-model code path (dummy models)."""
    import dummy_models

    models = tmp_path / "models"
    dummy_models.make_text_model(models / "text")
    dummy_models.make_clip_model(models / "clip")
    src, dest = workspace
    stats = _run(src, dest, models=models, ocr=False)
    assert stats["files copied"] > 50
    db = sqlite3.connect(str(dest / "_FileSorter" / "state.sqlite"))
    spaces = {r[0] for r in db.execute("SELECT DISTINCT tspace FROM files WHERE tspace IS NOT NULL")}
    vspaces = {r[0] for r in db.execute("SELECT DISTINCT vspace FROM files WHERE vspace IS NOT NULL")}
    labels = db.execute("SELECT COUNT(*) FROM files WHERE label IS NOT NULL").fetchone()[0]
    db.close()
    assert spaces == {"text:dummy-text"} and vspaces == {"visual:dummy-clip"} and labels > 0


def test_cancel_is_safe(workspace):
    import threading

    src, dest = workspace
    ev = threading.Event()
    calls = []

    def progress(stage, done, total, msg=""):
        calls.append(stage)
        if stage == "Reading files":
            ev.set()

    s = Settings(sources=[str(src)], destination=str(dest), ocr=False)
    stats = Engine(s, progress=progress, cancel=ev, models=dest.parent / "none").run()
    assert stats["cancelled"] == 1
    # the next run finishes the job
    stats = _run(src, dest, ocr=False)
    assert stats["cancelled"] == 0 and stats["files copied"] > 0


def test_destination_inside_source_is_skipped(workspace):
    src, _ = workspace
    dest = src / "Sorted"
    _run(src, dest, ocr=False)
    stats = _run(src, dest, ocr=False)
    assert stats["files copied"] == 0
