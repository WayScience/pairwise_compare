"""Spearman's rank correlation between pairs of rows."""

import numpy as np
import pandas as pd

from pairwise_compare.comparators.DotProductComparator import DotProductComparator, center_and_scale


class SpearmanCorrelation(DotProductComparator):
    """
    Spearman's rank correlation between the two rows of each pair, across their features.

    It is Pearson's correlation of the ranks of the features in each row, so it follows any increasing relation,
    not only a straight line, and ties get the average of their ranks.
    Higher is more similar: 1 is a perfect increasing relation, 0 is none, and -1 is a perfect decreasing relation.
    A row with the same value in every feature has no correlation, so its results are missing (NaN).

    Parameters
    ----------
    name:
        The name of the column with the correlations.
    """

    def __init__(self, name: str = "spearman_correlation"):
        """Name the column that holds the correlations."""
        super().__init__(name)

    def prepare(self, values: np.ndarray) -> np.ndarray:
        """Replace the features of every row with their ranks (ties get the average rank), and scale the rows as Pearson's do."""
        return center_and_scale(pd.DataFrame(values).rank(axis=1, method="average").to_numpy())
