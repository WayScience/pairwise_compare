"""Helpers of the tests: slow, simple ways to get the answers that the package computes."""

import numpy as np
import pandas as pd


def expected_order(df: pd.DataFrame, different: list[str], same: list[str] = ()) -> tuple[np.ndarray, np.ndarray]:
    """
    The pairs of rows (their positions in `df`) that a comparison should contain, in the order of its results.

    Two rows form a pair when they have a different value in every `different` column and the same value in every
    `same` column. The first row of a pair is the one with the smaller values in the group columns (the `same` columns
    and then the `different` columns, or the first `different` column and then the other ones without `same` columns).
    The pairs are ordered by the groups of the first row, of the second row, and then by the position of the rows.
    Every pair of rows is tested (with arrays of the size of rows x rows), so this is only for small dataframes
    without missing values in the group columns.

    Returns the positions of the first rows and of the second rows of the pairs, as two arrays.
    """

    allowed = np.triu(np.ones((len(df), len(df)), dtype=bool), k=1)  # each pair once: the first row before the second

    for columns, same_value in ((different, False), (same, True)):
        for column in columns:
            values = df[column].to_numpy()
            allowed &= (values[:, None] == values[None, :]) == same_value

    outer = list(same) if same else list(different[:1])
    inner = list(different) if same else list(different[1:])
    outer_keys = list(map(tuple, df[outer].to_numpy(dtype=object)))
    inner_keys = list(map(tuple, df[inner].to_numpy(dtype=object)))

    pairs = []
    for first, second in zip(*np.nonzero(allowed)):
        smaller_first = (outer_keys[first], inner_keys[first]) < (outer_keys[second], inner_keys[second])
        pairs.append((first, second) if smaller_first else (second, first))

    pairs.sort(key=lambda pair: (*(outer_keys[row] for row in pair), *(inner_keys[row] for row in pair), pair))
    rows0, rows1 = zip(*pairs) if pairs else ((), ())

    return np.array(rows0, dtype=int), np.array(rows1, dtype=int)


def numpy_correlations(df: pd.DataFrame, features: list[str], result: pd.DataFrame) -> np.ndarray:
    """The Pearson correlations of numpy for the pairs of wells in `result` (`well__group0` and `well__group1`)."""

    correlations = np.corrcoef(df[features].to_numpy(dtype=float))
    position = {well: row for row, well in enumerate(df["well"])}

    return np.array([correlations[position[a], position[b]] for a, b in zip(result["well__group0"], result["well__group1"])])
