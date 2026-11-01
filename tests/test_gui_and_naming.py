import time

import numpy as np
import pytest

from filesorter.organize import Terms, organize_documents


class FakeNamer:
    def __init__(self):
        self.calls = []

    def name_topic(self, keywords, examples):
        self.calls.append(("topic", keywords, examples))
        return f"Topic {keywords[0]}"

    def name_area(self, child_names, keywords):
        self.calls.append(("area", child_names))
        return "Area " + child_names[0].split()[-1]


def test_language_model_names_are_used():
    rng = np.random.default_rng(0)
    texts, X = [], []
    for words in (["invoice", "payment"], ["photo", "party"], ["excel", "training"]):
        c = rng.normal(size=16)
        for k in range(6):
            texts.append(f"{words[0]} {words[1]} document {k}")
            X.append(c + 0.1 * rng.normal(size=16))
    X = np.array(X)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    namer = FakeNamer()
    topics, _, _ = organize_documents(X, Terms.build(texts), texts, namer=namer, texts=texts)
    assert len(topics) == 3
    assert all(t.name.startswith("Topic ") for t in topics)
    assert all(t.area.startswith("Area ") for t in topics)
    assert any(c[0] == "topic" and c[2] for c in namer.calls)  # examples were given to the model


def test_window_runs_a_sort(sample, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from filesorter.config import Settings
    from filesorter.gui import MainWindow, _apply_palette

    monkeypatch.setenv("FILESORTER_MODELS", str(tmp_path / "no-models"))
    app = QApplication.instance() or QApplication([])
    _apply_palette(app)
    s = Settings(sources=[str(sample)], destination=str(tmp_path / "Sorted"), background=False, ocr=False)
    from PySide6.QtGui import QIcon

    win = MainWindow(s, QIcon())
    done = []
    win.bridge.finished.connect(lambda stats: done.append(stats))
    win.start.click()
    deadline = time.time() + 120
    while not done and time.time() < deadline:
        app.processEvents()
        time.sleep(0.05)
    app.processEvents()
    assert done and done[0]["files copied"] > 50
    assert "files copied" in win.stage.text()
    assert (tmp_path / "Sorted" / "Sorting report.html").exists()
    win.watcher.stop()
