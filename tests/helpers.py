"""Helpers of the tests: slow, simple ways to get the answers that the package computes."""

import numpy as np
import pandas as pd


def expected_pairs(df: pd.DataFrame, different: list[str], same: list[str] = ()) -> set[frozenset]:
    """
    All unordered pairs of wells that a comparison should contain.

    Two rows form a pair when they have a different value in every `different` column and the same value in every
    `same` column. Every pair of rows is tested (with arrays of the size of rows x rows), so this is only for small
    dataframes with a unique `well` column.
    """

    allowed = np.triu(np.ones((len(df), len(df)), dtype=bool), k=1)  # each pair once: the first row before the second

    for column in different:
        values = df[column].to_numpy()
        allowed &= values[:, None] != values[None, :]

    for column in same:
        values = df[column].to_numpy()
        allowed &= values[:, None] == values[None, :]

    wells = df["well"].to_numpy()
    first, second = np.nonzero(allowed)

    return {frozenset(pair) for pair in zip(wells[first], wells[second])}


def result_pairs(result: pd.DataFrame) -> set[frozenset]:
    """The unordered pairs of wells in the results, which must have `well__group0` and `well__group1` columns."""

    return {frozenset(pair) for pair in zip(result["well__group0"], result["well__group1"])}


def numpy_correlations(df: pd.DataFrame, features: list[str], result: pd.DataFrame) -> np.ndarray:
    """The Pearson correlations of numpy for the pairs of wells in `result` (`well__group0` and `well__group1`)."""

    correlations = np.corrcoef(df[features].to_numpy(dtype=float))
    position = {well: row for row, well in enumerate(df["well"])}

    return np.array([correlations[position[a], position[b]] for a, b in zip(result["well__group0"], result["well__group1"])])
