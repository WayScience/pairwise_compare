"""The template that every comparator follows."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Comparator(ABC):
    """
    Template of a comparator, which compares pairs of rows of a table of numbers.

    A comparator keeps no results: `compare` returns the values of the pairs it is given, so one comparator can be
    used for any number of comparisons.
    The engine first calls `prepare` once with the feature values of all the rows that can be compared, and then
    calls `compare` with the output of `prepare` and the row numbers of the pairs.
    The rows of a pair are always `rows0[k]` and `rows1[k]`, and the engine decides which rows are paired.

    Parameters
    ----------
    name:
        The name of the column of the results that holds the values of this comparator.
    """

    def __init__(self, name: str):
        """Store the name of the column of the results."""
        self.name = name

    def prepare(self, values: np.ndarray) -> Any:
        """
        Prepare the feature values once, so `compare` can use them for every pair (by default, they are used as they are).

        Parameters
        ----------
        values:
            The feature values as floats, with one row per sample and one column per feature.

        Returns
        -------
        Any
            Whatever `compare` needs, indexed by the same row numbers as `values`.
        """
        return values

    @abstractmethod
    def compare(self, prepared: Any, rows0: np.ndarray, rows1: np.ndarray) -> np.ndarray:
        """
        Compare the pairs of rows `(rows0[k], rows1[k])`.

        Parameters
        ----------
        prepared:
            The output of `prepare`.
        rows0, rows1:
            The row numbers (integer arrays of the same length) of the first and the second row of each pair.

        Returns
        -------
        np.ndarray
            One value for each pair, in the order of the pairs.
        """
