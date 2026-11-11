"""Cosine similarity between pairs of rows."""

import warnings

import numpy as np

from pairwise_compare.comparators.DotProductComparator import DotProductComparator


class CosineSimilarity(DotProductComparator):
    """
    Cosine similarity between the two rows of each pair: the cosine of the angle between the rows as vectors.

    Unlike Pearson's correlation, the features of a row are not centered, so rows with a large common value are similar.
    Higher is more similar: 1 means the same direction, 0 is perpendicular, and -1 is the opposite direction.
    A row with zero in every feature has no direction, so its results are missing (NaN).

    Parameters
    ----------
    name:
        The name of the column with the similarities.
    """

    def __init__(self, name: str = "cosine_similarity"):
        """Name the column that holds the similarities."""
        super().__init__(name)

    def prepare(self, values: np.ndarray) -> np.ndarray:
        """Scale every row to a length of 1, so the cosine similarity of two rows is the dot product of the rows."""

        lengths = np.linalg.norm(values, axis=1, keepdims=True)

        if (lengths == 0).any():
            warnings.warn(
                "A row has zero in every feature, so its cosine similarities are missing (NaN).",
                UserWarning,
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            return values / lengths
