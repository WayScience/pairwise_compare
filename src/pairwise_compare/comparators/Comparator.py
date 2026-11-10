"""The template that every comparator follows."""

from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd


class Comparator(ABC):
    """
    Template of a comparator, which compares the rows of two groups and keeps the results.

    A comparator stores its results in `self._comparisons`, which is later converted to a dataframe:
    each key is a column name, and the list of values of a key has one value per row of the dataframe.
    Call this constructor (`super().__init__()`) when inheriting from this class.

    The engine calls `__call__` for each pair of groups, and then `save_groups` to store the values of the
    group columns of the same pair, so `save_groups` can rely on the arguments of the call just before it.
    A comparator keeps the results of all the comparisons it was used for, so use a new comparator for each
    set of comparisons.
    """

    def __init__(self):
        """Start with no stored results."""
        self._comparisons = defaultdict(list)

    @property
    def comparisons(self) -> dict[str, list]:
        """The results, as a dictionary of column names to the list of values of that column."""
        return self._comparisons

    @abstractmethod
    def __call__(self, group0: pd.DataFrame, group1: pd.DataFrame) -> None:
        """
        Compare two groups and save the result in `self._comparisons`.

        Parameters
        ----------
        group0, group1:
            The feature values of the rows of each group, with one row per sample.
        """

    @abstractmethod
    def save_groups(self, group_columns: list[str], **groups: tuple) -> None:
        """
        Save the values of the group columns of the groups that were just compared.

        Parameters
        ----------
        group_columns:
            The names of the group columns to save.
        groups:
            The values of those columns for each group, as `group0=(value, ...)` and `group1=(value, ...)`,
            with one value per name in `group_columns`.
            Save them in columns named `<column>__<group name>`, for example `Metadata_Well__group0`.
        """
