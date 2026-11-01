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

    def name_area(self, child_names, keywords, avoid=None):
        self.calls.append(("area", child_names, avoid))
        if avoid is None and len(child_names) == 1:
            return child_names[0]        # lazy answer: same as the sub-folder -> must be asked again
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
    # a parent folder never repeats its only sub-folder's name
    assert all(t.area.lower() != t.name.lower() for t in topics)


def _topics(centres, per=8, noise=0.12, seed=0):
    rng = np.random.default_rng(seed)
    X = np.concatenate([c + noise * rng.normal(size=(per, len(c))) for c in centres])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_unrelated_topics_are_not_forced_into_one_area():
    from filesorter.organize import group_into_areas

    rng = np.random.default_rng(3)
    centres = np.linalg.qr(rng.normal(size=(32, 4)))[0].T      # 4 unrelated subjects
    X = _topics(centres)
    means = np.stack([X[i * 8:(i + 1) * 8].mean(0) for i in range(4)])
    from filesorter.organize import measure_levels

    labels = group_into_areas(means, measure_levels(X))
    assert len(set(labels)) == 4


def test_related_topics_share_an_area():
    from filesorter.organize import group_into_areas, measure_levels

    rng = np.random.default_rng(4)
    q = np.linalg.qr(rng.normal(size=(32, 6)))[0].T
    finance, people = q[0], q[1]
    centres = [finance + 0.9 * q[2], finance + 0.9 * q[3], people + 0.9 * q[4], people + 0.9 * q[5]]
    centres = [c / np.linalg.norm(c) for c in centres]
    X = _topics(centres)
    means = np.stack([X[i * 8:(i + 1) * 8].mean(0) for i in range(4)])
    labels = group_into_areas(means, measure_levels(X))
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]


def test_camera_photos_are_never_graphics():
    from filesorter.labels import LABELS, PHOTOS, SCANS, pick_label

    p = np.full(len(LABELS), 0.001)
    logo = next(i for i, lab in enumerate(LABELS) if lab[1] == "Logos")
    party = next(i for i, lab in enumerate(LABELS) if lab[1] == "Party & celebration")
    p[logo], p[party] = 0.6, 0.1
    p /= p.sum()
    assert LABELS[pick_label(p)[0]][1] == "Logos"
    j, _, area = pick_label(p, camera=True)
    assert area in (PHOTOS, SCANS) and LABELS[j][1] == "Party & celebration"


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


def test_clear_winner_rule_for_short_texts():
    from filesorter.organize import acceptable_match

    assert acceptable_match(np.array([0.9, 0.1])) == 0            # good match
    assert acceptable_match(np.array([0.4, 0.05, 0.0])) == 0      # modest, but clearly the best folder
    assert acceptable_match(np.array([0.4, 0.3])) is None          # modest and ambiguous
    assert acceptable_match(np.array([0.1, 0.0])) is None          # unrelated to everything


def test_parent_name_must_not_repeat_a_subfolder():
    from filesorter.organize import _too_close

    assert _too_close("Macros VBA", ["Excel Macros VBA"])
    assert _too_close("excel macros vba", ["Excel Macros VBA"])
    assert not _too_close("Training", ["Excel Macros VBA"])
    assert not _too_close("Human resources", ["Onboarding Interview Handbook"])


def test_short_text_joins_the_group_of_its_closest_files():
    """A file far from a group's average but very close to several of its members joins that group."""
    from filesorter.organize import Levels, neighbour_votes

    rng = np.random.default_rng(5)
    d = 24
    a, b = np.linalg.qr(rng.normal(size=(d, 2)))[0].T
    group_a = [a + 0.6 * rng.normal(size=d) for _ in range(8)]
    group_b = [b + 0.6 * rng.normal(size=d) for _ in range(8)]
    X = np.array(group_a + group_b + [group_a[0] + group_a[1] + group_a[2]])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    votes = neighbour_votes(X, [16], [list(range(8)), list(range(8, 16))], Levels(0.0, 0.8), 0.4)
    assert votes[0][0] == 0
