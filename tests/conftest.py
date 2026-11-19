import tempfile
import tomllib
from pathlib import Path

import pytest

GOLDEN = Path("dataEngine/data/test_data/src_golden")
OVERRIDES = tomllib.loads(Path("configs/data/test_data.toml").read_text(encoding="utf-8"))[
    "golden"
]["station_overrides"]
needs_golden = pytest.mark.skipif(not GOLDEN.is_dir(), reason="golden test data not present")


@pytest.fixture
def repo_tmp():
    """Temp folder inside the repo: records store paths relative to the data (same drive needed)."""
    Path("outputs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir="outputs") as folder:
        yield Path(folder)
