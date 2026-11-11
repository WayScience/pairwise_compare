"""The comparators whose value for a pair is the dot product of the two prepared rows."""

import warnings

import numpy as np

from pairwise_compare.comparators.Comparator import Comparator

# below this share of the possible pairs of the rows involved, picking the pairs one by one is faster than matrix products
_SPARSE_SHARE = 1 / 400


def center_and_scale(values: np.ndarray) -> np.ndarray:
    """
    Center every row and scale it to a length of 1, so the correlation of two rows is the dot product of the rows.

    A row with the same value in every feature has no correlation, so it becomes missing (NaN), with a warning.
    """

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


class DotProductComparator(Comparator):
    """
    Base of the comparators whose value for a pair is the dot product of the two rows that `prepare` returns.

    A subclass only needs `prepare`, which gives the rows (for example, centered and of length 1, so that the
    dot product is the correlation). `compare` takes the dot products by matrix products when many of the possible
    pairs are needed, and pair by pair when few are, in blocks that keep the memory use low.
    """

    def compare(self, prepared: np.ndarray, rows0: np.ndarray, rows1: np.ndarray) -> np.ndarray:
        """The dot product of the two rows of each pair, from matrix products when many of the possible pairs are needed."""

        n_pairs = len(rows0)
        rows = np.unique(np.concatenate([rows0, rows1]))

        if n_pairs < _SPARSE_SHARE * len(rows) ** 2:
            return self._gather(prepared, rows0, rows1)

        return self._blocks(prepared[rows], np.searchsorted(rows, rows0), np.searchsorted(rows, rows1))

    def _gather(self, prepared: np.ndarray, rows0: np.ndarray, rows1: np.ndarray) -> np.ndarray:
        """Dot product of the two rows of each pair, a chunk of pairs at a time."""

        step = self._block_size(prepared.shape[1])
        values = np.empty(len(rows0))

        for start in range(0, len(rows0), step):
            chunk = slice(start, start + step)
            values[chunk] = np.einsum("ij,ij->i", prepared[rows0[chunk]], prepared[rows1[chunk]])

        return values

    def _blocks(self, prepared: np.ndarray, rows0: np.ndarray, rows1: np.ndarray) -> np.ndarray:
        """The dot products of a block of rows with all rows by a matrix product, for the pairs that start in the block."""

        order = np.argsort(rows0, kind="stable")
        sorted_rows0, sorted_rows1 = rows0[order], rows1[order]
        step = self._block_size(len(prepared))
        values = np.empty(len(rows0))

        for start in range(0, len(prepared), step):
            low, high = np.searchsorted(sorted_rows0, [start, start + step])
            if high > low:
                block = prepared[start : start + step] @ prepared.T
                values[order[low:high]] = block[sorted_rows0[low:high] - start, sorted_rows1[low:high]]

        return values
