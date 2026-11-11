"""Distances between pairs of rows: Euclidean and Manhattan."""

from abc import abstractmethod

import numpy as np

from pairwise_compare.comparators.Comparator import Comparator


class PairwiseDistance(Comparator):
    """
    Base of the comparators whose value for a pair is a distance between the two rows.

    A distance is lower for more similar rows (0 for identical rows), and is missing (NaN) when a row has a missing value.
    It is computed exactly from the differences of the features, a chunk of pairs at a time, which keeps the memory use low.
    A subclass gives `distance` of the differences of the rows of a chunk of pairs.
    """

    def compare(self, prepared: np.ndarray, rows0: np.ndarray, rows1: np.ndarray) -> np.ndarray:
        """The distance between the two rows of each pair, a chunk of pairs at a time."""

        step = self._block_size(prepared.shape[1])
        distances = np.empty(len(rows0))

        for start in range(0, len(rows0), step):
            chunk = slice(start, start + step)
            distances[chunk] = self.distance(prepared[rows0[chunk]] - prepared[rows1[chunk]])

        return distances

    @abstractmethod
    def distance(self, differences: np.ndarray) -> np.ndarray:
        """The distance for each row of `differences`, which are the differences of the features of the pairs of rows."""


class EuclideanDistance(PairwiseDistance):
    """
    Euclidean distance between the two rows of each pair: the square root of the sum of the squared differences of the features.

    Lower is more similar, and 0 means identical rows. The features are used as they are, so features with large
    values weigh more (scale the features first if they should weigh the same).

    Parameters
    ----------
    name:
        The name of the column with the distances.
    """

    def __init__(self, name: str = "euclidean_distance"):
        """Name the column that holds the distances."""
        super().__init__(name)

    def distance(self, differences: np.ndarray) -> np.ndarray:
        """The square root of the sum of the squared differences of each row."""
        return np.sqrt((differences * differences).sum(axis=1))


class ManhattanDistance(PairwiseDistance):
    """
    Manhattan (city block) distance between the two rows of each pair: the sum of the absolute differences of the features.

    Lower is more similar, and 0 means identical rows. The features are used as they are, so features with large
    values weigh more (scale the features first if they should weigh the same).

    Parameters
    ----------
    name:
        The name of the column with the distances.
    """

    def __init__(self, name: str = "manhattan_distance"):
        """Name the column that holds the distances."""
        super().__init__(name)

    def distance(self, differences: np.ndarray) -> np.ndarray:
        """The sum of the absolute differences of each row."""
        return np.abs(differences).sum(axis=1)
