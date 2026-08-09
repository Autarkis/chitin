import tomllib
from pathlib import Path

import chitin


def test_version_matches_pyproject():
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    assert chitin.__version__ == data["project"]["version"]
