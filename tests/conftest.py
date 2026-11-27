import tempfile
import tomllib
from pathlib import Path

import pytest

import tracking
from dataEngine.__main__ import read_overrides

GOLDEN = Path("dataEngine/data/test_data/src_golden")
OVERRIDES = read_overrides(
    tomllib.loads(Path("configs/data/test_data.toml").read_text(encoding="utf-8"))["golden"]
)
needs_golden = pytest.mark.skipif(not GOLDEN.is_dir(), reason="golden test data not present")


@pytest.fixture(autouse=True)
def no_trackers(monkeypatch):
    """Tests never read the repo's .env or send anything to W&B / MLflow."""
    monkeypatch.setattr(tracking, "load_env", lambda path=None: None)
    for variable in tracking.TRACKERS.values():
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def repo_tmp():
    """Temp folder inside the repo: records store paths relative to the data (same drive needed)."""
    Path("outputs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir="outputs") as folder:
        yield Path(folder)
