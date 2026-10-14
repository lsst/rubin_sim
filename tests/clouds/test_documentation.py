"""Execute the published clouds examples during ordinary pytest collection."""

import doctest
import importlib
from pathlib import Path

import matplotlib
import pytest
from matplotlib import pyplot as plt

matplotlib.use("Agg")

DOCS = Path(__file__).resolve().parents[2] / "docs"


@pytest.mark.parametrize("page", ["clouds.rst", "clouds-api.rst", "clouds-architecture.rst"])
def test_documentation_pages(page, tmp_path, monkeypatch):
    """Read real page sources without hidden inputs or copied examples."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    path = DOCS / page
    text = path.read_text(encoding="utf-8")
    examples = doctest.DocTestParser().get_examples(text)
    if page == "clouds.rst":
        assert examples, "The executable user guide must not silently test zero examples"
    test = doctest.DocTestParser().get_doctest(text, {}, page, str(path), 0)
    runner = doctest.DocTestRunner()
    try:
        runner.run(test, clear_globs=False)
        result = runner.summarize()
        assert result.failed == 0
        assert result.attempted == len(examples), "Every executable example must run, without skips"
        assert not plt.get_fignums(), "Documentation examples must close their figures"
        assert not list(tmp_path.iterdir()), "Documentation examples must clean up their files"
    finally:
        workspace = test.globs.get("workspace")
        if workspace is not None:
            workspace.cleanup()
        plt.close("all")


@pytest.mark.parametrize(
    "module_name", ["rubin_sim.clouds", "rubin_sim.clouds.data", "rubin_sim.clouds.generate"]
)
def test_documentation_docstrings(module_name, tmp_path, monkeypatch):
    """Cover source docstrings behind imported API members.

    These modules currently have no executable docstring examples; unlike the
    user guide, zero examples here is intentional. Do not import ``__main__``,
    which executes the CLI rather than defining API members.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MPLBACKEND", "Agg")
    module = importlib.import_module(module_name)
    tests = doctest.DocTestFinder().find(module)
    runner = doctest.DocTestRunner()
    try:
        for test in tests:
            runner.run(test)
        result = runner.summarize()
        assert result.failed == 0
        assert result.attempted == sum(len(test.examples) for test in tests)
        assert not plt.get_fignums(), "API examples must close their figures"
        assert not list(tmp_path.iterdir()), "API examples must clean up their files"
    finally:
        plt.close("all")
