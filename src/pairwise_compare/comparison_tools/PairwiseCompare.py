"""Compare the rows of a dataframe in pairs, within groups or between groups."""

import warnings
from collections.abc import Sequence
from typing import Optional

import numpy as np
import pandas as pd

from pairwise_compare.comparators.Comparator import Comparator

# the most rows of a block of the pairs of all rows (a block holds this many pairs of rows, as booleans)
_PAIR_BLOCK = 20_000_000


def as_column_list(name: str, columns: Optional[Sequence[str]]) -> list[str]:
    """
    Check that `columns` is a list or tuple of column names, and return it as a list without duplicates.

    `None` gives an empty list. The names stay in the order they were given.
    """

    if columns is None:
        return []

    if isinstance(columns, str):
        raise TypeError(f"`{name}` must be a list of column names, not a single string ('{columns}').")

    if not isinstance(columns, (list, tuple)):
        raise TypeError(f"`{name}` must be a list or tuple of column names, not {type(columns).__name__}.")

    if any(not isinstance(column, str) for column in columns):
        raise TypeError(f"`{name}` must only contain column names (strings).")

    return list(dict.fromkeys(columns))


def _codes(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """The values of each column as integers that have the order of the values (-1 for a missing value)."""

    return np.column_stack([pd.factorize(df[column], sort=True)[0] for column in columns])


def _ranks(codes: np.ndarray) -> np.ndarray:
    """The rank of the combination of values in each row of `codes`, ordered by the first column, then the second, ..."""

    if codes.shape[1] == 1:
        return codes[:, 0]

    return np.unique(codes, axis=0, return_inverse=True)[1].reshape(-1)


def _sort_order(keys: list[np.ndarray]) -> np.ndarray:
    """
    The order that sorts by the first array of `keys`, then by the second, and so on (the arrays are non-negative integers).

    When the keys fit in 63 bits together they are combined into one number, which sorts much faster than several keys.
    """

    bits = [int(key.max()).bit_length() if len(key) else 0 for key in keys]

    if sum(bits) > 63:
        return np.lexsort(keys[::-1])

    combined = np.zeros(len(keys[0]), dtype=np.int64)
    for key, width in zip(keys, bits):
        combined = (combined << width) | key

    return np.argsort(combined)


class PairwiseCompare:
    """
    Compare the rows of a dataframe in pairs, with the rows split into groups twice.

    The rows are first split into *outer* groups by their values in `outer_columns`, and the rows of each outer
    group are then split into *inner* groups by their values in `inner_columns`.
    The rows of two inner groups are compared with the comparator, as long as the two groups have a different value
    in every inner column. Each pair of rows is compared once, in one direction only.

    - `intra_comparisons` compares rows of inner groups of the same outer group.
    - `inter_comparisons` compares rows of inner groups of two outer groups that have a different value in every outer column.

    The groups are sorted by their values, and the first group of a pair is the group that comes first:
    the pairs are in the order of the outer group of the first row and then of the second row, then of the inner group
    of the first row and of the second row, and then of the position of the rows in the dataframe.

    Users almost always want `pairwise_compare.compare_pairs` instead, which checks how the columns are combined.

    Parameters
    ----------
    df:
        The rows to compare, with the group columns and the feature columns.
    comparator:
        Defines how two rows are compared. It keeps no results, so it can be used again.
    outer_columns:
        The columns that define the outer groups.
    inner_columns:
        The columns that define the inner groups.
    features:
        The numeric feature columns to compare.
    drop:
        Group columns (outer or inner) whose values are left out of the results, although they define the groups.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        comparator: Comparator,
        outer_columns: list[str],
        inner_columns: list[str],
        features: list[str],
        drop: Optional[list[str]] = None,
    ):
        """Check the arguments and prepare the groups and the values (the class docstring describes the parameters)."""

        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a pandas DataFrame.")

        if not isinstance(comparator, Comparator):
            raise TypeError("`comparator` must be an instance of Comparator.")

        outer_columns = as_column_list("outer_columns", outer_columns)
        inner_columns = as_column_list("inner_columns", inner_columns)
        features = as_column_list("features", features)
        drop = as_column_list("drop", drop)

        if not outer_columns or not inner_columns:
            raise ValueError("`outer_columns` and `inner_columns` must each have at least one column.")

        if df.empty:
            raise ValueError("`df` is empty.")

        missing = (set(outer_columns) | set(inner_columns) | set(features) | set(drop)) - set(df.columns)
        if missing:
            raise ValueError(f"The dataframe does not have the columns {sorted(missing)}.")

        not_numeric = [column for column in features if not pd.api.types.is_numeric_dtype(df[column].dtype)]
        if not_numeric:
            raise TypeError(f"The feature columns {not_numeric} are not numeric.")

        self._warn_about_missing_values(df, outer_columns + inner_columns, features)

        # Only the rows without a missing group value can be compared. From here on, a row is its number among those rows.
        has_groups = ~df[outer_columns + inner_columns].isna().any(axis=1).to_numpy()
        outer_codes = _codes(df, outer_columns)[has_groups]
        inner_codes = _codes(df, inner_columns)[has_groups]

        self._comparator = comparator
        self._groups = df[outer_columns + inner_columns].iloc[np.flatnonzero(has_groups)].reset_index(drop=True)
        self._values = df[features].to_numpy(dtype=float, na_value=np.nan)[has_groups]
        self._outer_codes, self._inner_codes = outer_codes, inner_codes
        self._outer_rank, self._inner_rank = _ranks(outer_codes), _ranks(inner_codes)

        # the group columns to save with the results
        self._saved_columns = [column for column in outer_columns + inner_columns if column not in drop]

    @staticmethod
    def _warn_about_missing_values(df: pd.DataFrame, group_columns: list[str], features: list[str]) -> None:
        """Warn about missing values in the columns that are used. Missing values elsewhere are not a problem."""

        missing_groups = df[group_columns].isna()
        if missing_groups.any(axis=None):
            affected = missing_groups.columns[missing_groups.any()].tolist()
            warnings.warn(
                f"{int(missing_groups.any(axis=1).sum())} of {len(df)} rows have a missing value in the group "
                f"columns {affected}. Rows with a missing group value are left out of the comparisons.",
                UserWarning,
            )

        missing_features = df[features].isna().any(axis=1)
        if missing_features.any():
            warnings.warn(
                f"{int(missing_features.sum())} of {len(df)} rows have a missing value in the feature columns. "
                "The results of every comparison with these rows are missing.",
                UserWarning,
            )

    def intra_comparisons(self) -> pd.DataFrame:
        """Compare the rows of different inner groups of the same outer group, and return the results."""

        counts = pd.Series(self._inner_rank).groupby(self._outer_rank).nunique()
        if (counts < 2).any():
            warnings.warn(
                f"{int((counts < 2).sum())} of {len(counts)} outer groups have fewer than two inner groups, "
                "so nothing is compared in them.",
                UserWarning,
            )

        # all the pairs of rows of an outer group, with a different value in every inner column
        frame = pd.DataFrame(self._inner_codes, columns=[f"inner{k}" for k in range(self._inner_codes.shape[1])])
        frame["outer"], frame["row"] = self._outer_rank, np.arange(len(frame))
        pairs = frame.merge(frame, on="outer", suffixes=("_0", "_1"))
        keep = pairs["row_0"] < pairs["row_1"]
        for column in frame.columns[: self._inner_codes.shape[1]]:
            keep &= pairs[f"{column}_0"] != pairs[f"{column}_1"]
        pairs = pairs[keep]

        return self._results(pairs["row_0"].to_numpy(), pairs["row_1"].to_numpy(), split_by_outer_group=True)

    def inter_comparisons(self) -> pd.DataFrame:
        """Compare the rows of different inner groups of outer groups that differ in every outer column, and return the results."""

        codes = np.column_stack([self._outer_codes, self._inner_codes])
        n_rows = len(codes)
        step = max(1, _PAIR_BLOCK // max(n_rows, 1))
        first, second = [np.empty(0, dtype=np.intp)], [np.empty(0, dtype=np.intp)]  # empty when no row is left

        # the pairs of all rows, a block of rows at a time
        for start in range(0, n_rows, step):
            rows = np.arange(start, min(start + step, n_rows))
            later = np.arange(n_rows)[None, :] > rows[:, None]
            for column in range(codes.shape[1]):
                later &= codes[rows, column][:, None] != codes[None, :, column]
            block_first, block_second = np.nonzero(later)
            first.append(block_first + start)
            second.append(block_second)

        return self._results(np.concatenate(first), np.concatenate(second), split_by_outer_group=False)

    def _results(self, first: np.ndarray, second: np.ndarray, split_by_outer_group: bool) -> pd.DataFrame:
        """Order and compare the pairs of rows (`first[k]`, `second[k]`), and return the table of the results."""

        outer, inner = self._outer_rank, self._inner_rank

        # the first row of a pair is in the group that comes first, by outer group and then by inner group
        swap = (outer[first] > outer[second]) | ((outer[first] == outer[second]) & (inner[first] > inner[second]))
        rows0, rows1 = np.where(swap, second, first), np.where(swap, first, second)
        order = _sort_order([outer[rows0], outer[rows1], inner[rows0], inner[rows1], rows0, rows1])
        rows0, rows1 = rows0[order], rows1[order]

        prepared = self._comparator.prepare(self._values)

        # within outer groups, the pairs of an outer group are together and are compared as a set
        edges = [0, len(rows0)]
        if split_by_outer_group and len(rows0):
            edges = [0, *(np.flatnonzero(np.diff(outer[rows0])) + 1), len(rows0)]

        chunks = [self._comparator.compare(prepared, rows0[a:b], rows1[a:b]) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        values = np.concatenate(chunks) if chunks else np.empty(0)

        results = {self._comparator.name: values}
        for column in self._saved_columns:
            for group, rows in enumerate((rows0, rows1)):
                results[f"{column}__group{group}"] = self._groups[column].take(rows).reset_index(drop=True)

        return pd.DataFrame(results)
