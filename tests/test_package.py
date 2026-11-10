"""Tests of what the package offers to import."""

import importlib
import importlib.metadata

import pairwise_compare
from pairwise_compare import Comparator, PearsonsCorrelation


def test_public_names_and_version():
    """Public names and version."""
    assert set(pairwise_compare.__all__) == {"Comparator", "PearsonsCorrelation", "compare_pairs"}
    assert issubclass(PearsonsCorrelation, Comparator)
    assert isinstance(pairwise_compare.__version__, str)


def test_version_when_the_package_is_not_installed(monkeypatch):
    """Version when the package is not installed."""
    def not_installed(name):
        """Stand in for a package version lookup that fails."""
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", not_installed)
    try:
        assert importlib.reload(pairwise_compare).__version__ == "unknown"
    finally:
        monkeypatch.undo()
        importlib.reload(pairwise_compare)
