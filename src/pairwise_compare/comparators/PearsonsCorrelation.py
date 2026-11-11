"""Pearson's correlation between pairs of rows."""

import numpy as np

from pairwise_compare.comparators.DotProductComparator import DotProductComparator, center_and_scale


class PearsonsCorrelation(DotProductComparator):
    """
    Pearson's correlation between the two rows of each pair, across their features.

    Higher is more similar: 1 is a perfect positive correlation, 0 is none, and -1 is a perfect negative correlation.
    A row with the same value in every feature has no correlation, so its results are missing (NaN).

    Parameters
    ----------
    name:
        The name of the column with the correlations.
    """

    def __init__(self, name: str = "pearsons_correlation"):
        """Name the column that holds the correlations."""
        super().__init__(name)

    def prepare(self, values: np.ndarray) -> np.ndarray:
        """Center every row and scale it to a length of 1, so the correlation of two rows is the dot product of the rows."""
        return center_and_scale(values)
