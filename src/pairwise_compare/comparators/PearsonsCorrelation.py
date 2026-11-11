"""Pearson's correlation between pairs of rows."""

import warnings

import numpy as np

from pairwise_compare.comparators.Comparator import Comparator

# the most numbers that a block of correlations or a chunk of gathered rows holds, which keeps the memory use low
_BLOCK_ELEMENTS = 10_000_000

# below this share of the possible pairs of the rows involved, picking the pairs one by one is faster than matrix products
_SPARSE_SHARE = 1 / 400


class PearsonsCorrelation(Comparator):
    """
    Pearson's correlation between the two rows of each pair, across their features.

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

        centered = values - values.mean(axis=1, keepdims=True)

        # A row with the same value in every feature is found from the raw values, because centering can leave
        # a rounding error (for [0.1, 0.1, 0.1] the mean is 0.10000000000000002).
        same_value = np.ptp(values, axis=1) == 0
        centered[same_value] = 0

        if same_value.any():
            warnings.warn(
                "A row has the same value in every feature, so its correlations are missing (NaN).",
                UserWarning,
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            return centered / np.linalg.norm(centered, axis=1, keepdims=True)

    def compare(self, prepared: np.ndarray, rows0: np.ndarray, rows1: np.ndarray) -> np.ndarray:
        """The correlation of each pair, from matrix products when many of the possible pairs are needed."""

        n_pairs = len(rows0)
        rows = np.unique(np.concatenate([rows0, rows1]))

        if n_pairs < _SPARSE_SHARE * len(rows) ** 2:
            return self._gather(prepared, rows0, rows1)

        return self._blocks(prepared[rows], np.searchsorted(rows, rows0), np.searchsorted(rows, rows1))

    @staticmethod
    def _gather(prepared: np.ndarray, rows0: np.ndarray, rows1: np.ndarray) -> np.ndarray:
        """Dot product of the two rows of each pair, a chunk of pairs at a time."""

        step = max(1, _BLOCK_ELEMENTS // prepared.shape[1])
        correlations = np.empty(len(rows0))

        for start in range(0, len(rows0), step):
            chunk = slice(start, start + step)
            correlations[chunk] = np.einsum("ij,ij->i", prepared[rows0[chunk]], prepared[rows1[chunk]])

        return correlations

    @staticmethod
    def _blocks(prepared: np.ndarray, rows0: np.ndarray, rows1: np.ndarray) -> np.ndarray:
        """The correlations of a block of rows with all rows by a matrix product, for the pairs that start in the block."""

        order = np.argsort(rows0, kind="stable")
        sorted_rows0, sorted_rows1 = rows0[order], rows1[order]
        step = max(1, _BLOCK_ELEMENTS // len(prepared))
        correlations = np.empty(len(rows0))

        for start in range(0, len(prepared), step):
            low, high = np.searchsorted(sorted_rows0, [start, start + step])
            if high > low:
                block = prepared[start : start + step] @ prepared.T
                correlations[order[low:high]] = block[sorted_rows0[low:high] - start, sorted_rows1[low:high]]

        return correlations
