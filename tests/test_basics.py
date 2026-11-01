import socket
import urllib.request

import numpy as np
import pytest
from PIL import Image, ImageDraw

from filesorter import offline
from filesorter.features import image_hashes, text_fingerprint
from filesorter.organize import cluster, name_from_keywords, segment_events
from filesorter.textutil import looks_like_copy, safe_name, surface_form, terms_and_phrases, unique_name


def test_offline_guard_blocks_internet_but_not_loopback():
    offline.enable()
    with pytest.raises(OSError):
        urllib.request.urlopen("http://example.com", timeout=2)
    with pytest.raises(OSError):
        socket.create_connection(("8.8.8.8", 53), timeout=2)
    assert offline.verify()
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    c = socket.create_connection(srv.getsockname(), timeout=2)  # loopback still works
    c.close()
    srv.close()


@pytest.mark.parametrize("raw,expected", [
    ('Budget: 2025 / "final"?', "Budget 2025 final"),
    ("CON", "_CON"),
    ("trailing dots...", "trailing dots"),
    ("", "Untitled"),
])
def test_safe_name(raw, expected):
    assert safe_name(raw) == expected


def test_unique_name_case_insensitive():
    assert unique_name("Photos", {"photos"}) == "Photos (2)"
    assert unique_name("Photos", {"photos", "Photos (2)"}) == "Photos (3)"


def test_copy_markers():
    assert looks_like_copy("Rapport (2).docx")
    assert looks_like_copy("Rapport - Copie.docx")
    assert looks_like_copy("Copy of budget.xlsx")
    assert not looks_like_copy("Rapport final.docx")


def test_surface_form_keeps_acronyms_and_apostrophes():
    assert surface_form("vpn access", "Set up VPN access today. VPN access for all.") == "VPN access"
    assert surface_form("chiffre d affaires", "Le chiffre d’affaires 2024") == "chiffre d'affaires"


def test_phrases():
    t = terms_and_phrases("La note de frais doit être validée. Note de frais de juin.")
    assert "note de frais" in t and "frais" in t
    assert "de" not in t


def test_name_from_keywords():
    sample = "Plafond de dépenses et indemnités kilométriques"
    assert name_from_keywords(["plafond de dépenses", "indemnités kilométriques"], "x", sample) == \
        "Plafond de dépenses & Indemnités kilométriques"
    assert name_from_keywords([], "Misc") == "Misc"


def _picture(seed, size=(800, 600)):
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", size, tuple(int(x) for x in rng.integers(0, 255, 3)))
    d = ImageDraw.Draw(img)
    for _ in range(30):
        x, y = rng.integers(0, size[0]), rng.integers(0, size[1])
        r = int(rng.integers(20, 120))
        d.ellipse([x - r, y - r, x + r, y + r], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
    return img


def test_perceptual_hash_survives_resize_but_separates_pictures():
    a = _picture(1)
    b = a.resize((400, 300))
    c = _picture(2)
    ha, hb, hc = image_hashes(a), image_hashes(b), image_hashes(c)
    ham = lambda x, y: bin((x ^ y) & (2 ** 64 - 1)).count("1")  # noqa: E731
    assert ham(ha[0], hb[0]) <= 4
    assert ham(ha[0], hc[0]) > 12


def test_text_fingerprint_ignores_layout():
    t = "Contrat de prestation entre la société Alpha et la société Beta. " * 10
    assert text_fingerprint(t) == text_fingerprint(t.replace(" ", "\n").upper())
    assert text_fingerprint("short") is None


def test_cluster_finds_groups():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(4, 32))
    X = np.concatenate([c + 0.15 * rng.normal(size=(15, 32)) for c in centers])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    groups, scores, _, _ = cluster(X, "normal")
    assert len(groups) == 4
    assert sorted(len(g) for g in groups) == [15, 15, 15, 15]
    for g in groups:  # each group is pure
        assert len({i // 15 for i in g}) == 1


def test_event_segmentation():
    h = 3600
    times = [0, h, 2 * h, 30 * h, 31 * h, 100 * h]
    assert segment_events(times) == [[0, 1, 2], [3, 4], [5]]


def test_boilerplate_file_names_are_not_used_as_content():
    from filesorter.engine import path_context

    assert path_context("Pictures/Screenshots/Capture d'écran 2024-05-03.png") == ""
    assert path_context("Pictures/Screenshot 2024-05-02 101500.png") == ""
    assert path_context("DCIM/IMG_4100.JPG") == ""
    assert path_context("Projets/Formation Excel/Budget prévisionnel 2025.xlsx") == \
        "Projets Formation Excel Budget prévisionnel 2025"
