"""Check the documented clouds architecture during ordinary pytest."""

import doctest
from pathlib import Path


def test_clouds_architecture():
    """Execute the architecture page itself, not copies of its examples."""
    path = Path(__file__).resolve().parents[2] / "docs" / "clouds-architecture.rst"
    result = doctest.testfile(str(path), module_relative=False, encoding="utf-8")
    assert result.failed == 0
    assert result.attempted > 0, "The architecture page must contain executable examples"
