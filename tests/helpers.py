"""Helpers of the tests: slow, simple ways to get the answers that the package computes."""

from itertools import combinations

import numpy as np
import pandas as pd

# the feature columns of the `small_df` fixture
FEATURES = ["f1", "f2", "f3"]


def expected_pairs(df: pd.DataFrame, different: list[str], same: list[str] = ()) -> set[frozenset]:
    """
    All unordered pairs of wells that a comparison should contain.

    Two rows form a pair when they have a different value in every `different` column and the same value in every
    `same` column. This loops over every pair of rows, so it is only for small dataframes with a unique `well` column.
    """

    pairs = set()

    for first, second in combinations(range(len(df)), 2):
        row0, row1 = df.iloc[first], df.iloc[second]

        if all(row0[c] != row1[c] for c in different) and all(row0[c] == row1[c] for c in same):
            pairs.add(frozenset((row0["well"], row1["well"])))

    return pairs


def result_pairs(result: pd.DataFrame) -> set[frozenset]:
    """The unordered pairs of wells in the results, which must have `well__group0` and `well__group1` columns."""

    return {frozenset(pair) for pair in zip(result["well__group0"], result["well__group1"])}


def correlation_of_wells(df: pd.DataFrame, features: list[str], well0: str, well1: str) -> float:
    """The Pearson correlation of two wells across the features, from numpy."""

    by_well = df.set_index("well")[features]
    return float(np.corrcoef(by_well.loc[well0], by_well.loc[well1])[0, 1])
