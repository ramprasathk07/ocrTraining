import tempfile
from pathlib import Path

import pytest

GOLDEN = Path("dataEngine/data")
needs_golden = pytest.mark.skipif(not GOLDEN.is_dir(), reason="golden sample data not present")


@pytest.fixture
def repo_tmp():
    """Temp folder inside the repo: records store paths relative to the data (same drive needed)."""
    Path("outputs").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir="outputs") as folder:
        yield Path(folder)
