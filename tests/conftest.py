import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _isolated_profile(tmp_path, monkeypatch):
    """Keep settings/logs of tests out of the real user profile."""
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "localappdata"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))


@pytest.fixture(scope="session")
def sample(tmp_path_factory):
    import make_sample

    return make_sample.build(tmp_path_factory.mktemp("sample") / "src")
