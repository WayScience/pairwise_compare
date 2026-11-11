"""
Compare the rows of a tidy pandas dataframe in pairs.

A tidy dataframe has one row per sample (for example, one well of an experiment) and one column per variable.
Some columns are metadata that define the groups to compare (for example, the cell line or the plate),
and the other columns are numeric features (for example, measurements of the cell morphology).
See `compare_pairs` for the main function.
"""

from importlib.metadata import PackageNotFoundError, version

from pairwise_compare.comparators.Comparator import Comparator
from pairwise_compare.comparators.CosineSimilarity import CosineSimilarity
from pairwise_compare.comparators.Distances import EuclideanDistance, ManhattanDistance
from pairwise_compare.comparators.PearsonsCorrelation import PearsonsCorrelation
from pairwise_compare.comparators.SpearmanCorrelation import SpearmanCorrelation
from pairwise_compare.comparison_tools.PairwiseCompareManager import compare_pairs

try:
    __version__ = version("pairwise-compare")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "unknown"

__all__ = [
    "Comparator",
    "CosineSimilarity",
    "EuclideanDistance",
    "ManhattanDistance",
    "PearsonsCorrelation",
    "SpearmanCorrelation",
    "compare_pairs",
]
