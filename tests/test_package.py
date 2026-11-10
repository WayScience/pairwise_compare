"""Tests of what the package offers to import."""

import pairwise_compare
from pairwise_compare import Comparator, PearsonsCorrelation, compare_pairs


def test_public_names():
    assert set(pairwise_compare.__all__) == {"Comparator", "PearsonsCorrelation", "compare_pairs"}
    assert issubclass(PearsonsCorrelation, Comparator)
    assert callable(compare_pairs)


def test_version_is_a_string():
    assert isinstance(pairwise_compare.__version__, str)


def test_version_when_the_package_is_not_installed(monkeypatch):
    import importlib
    import importlib.metadata

    def not_installed(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", not_installed)
    try:
        assert importlib.reload(pairwise_compare).__version__ == "unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(pairwise_compare)
