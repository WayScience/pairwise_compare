"""Pearson's correlation between the rows of two groups."""

import warnings

import numpy as np
import pandas as pd

from pairwise_compare.comparators.Comparator import Comparator


class PearsonsCorrelation(Comparator):
    """
    Pearson's correlation between every row of one group and every row of another group.

    The correlation of two rows is computed across their features.
    A row with the same value in every feature has no correlation, so its results are missing (NaN).

    Parameters
    ----------
    name:
        The name of the column with the correlations.
    """

    def __init__(self, name: str = "pearsons_correlation"):
        super().__init__()
        self._name = name
        self._n_comparisons = 0

    def __call__(self, group0: pd.DataFrame, group1: pd.DataFrame) -> None:
        """Save the correlation of each row of `group0` with each row of `group1` (one group0 row after another)."""

        values0 = group0.to_numpy(dtype=float, na_value=np.nan)
        values1 = group1.to_numpy(dtype=float, na_value=np.nan)
        self._n_comparisons = len(values0) * len(values1)

        centered0 = values0 - values0.mean(axis=1, keepdims=True)
        centered1 = values1 - values1.mean(axis=1, keepdims=True)
        norms0 = np.linalg.norm(centered0, axis=1, keepdims=True)
        norms1 = np.linalg.norm(centered1, axis=1, keepdims=True)

        if (norms0 == 0).any() or (norms1 == 0).any():
            warnings.warn(
                "A row has the same value in every feature, so its correlations are missing (NaN).",
                UserWarning,
            )

        with np.errstate(divide="ignore", invalid="ignore"):
            correlations = np.dot(centered0, centered1.T) / np.dot(norms0, norms1.T)

        self._comparisons[self._name].extend(correlations.ravel().tolist())

    def save_groups(self, group_columns: list[str], **groups: tuple) -> None:
        """Repeat the group values once for each correlation of the last comparison."""

        for position, column in enumerate(group_columns):
            for group_name, group_values in groups.items():
                self._comparisons[f"{column}__{group_name}"].extend(
                    [group_values[position]] * self._n_comparisons
                )
