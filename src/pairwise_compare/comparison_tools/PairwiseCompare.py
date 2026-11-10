"""Compare the rows of a dataframe in pairs, within groups or between groups."""

import warnings
from collections.abc import Sequence
from itertools import combinations, product
from typing import Optional

import numpy as np
import pandas as pd

from pairwise_compare.comparators.Comparator import Comparator


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


def _group_rows(
    df: pd.DataFrame, columns: list[str], rows: Optional[np.ndarray] = None
) -> dict[tuple, np.ndarray]:
    """
    Group the rows of `df` by their values in `columns`.

    Returns a dictionary from the tuple of values of a group to the row positions of that group in `df`.
    Only the rows at the positions `rows` are grouped when they are given.
    The groups are sorted by their values, and rows with a missing value in `columns` are left out.
    """

    subset = df if rows is None else df.iloc[rows]
    groups = {}

    for key, positions in subset.groupby(columns, sort=True).indices.items():
        # grouping by a single column gives a value, and by several columns a tuple
        key = key if isinstance(key, tuple) else (key,)
        groups[key] = positions if rows is None else rows[positions]

    return groups


def _describe(key: tuple) -> str:
    """The values of a group, for the warnings."""

    return ", ".join(str(value) for value in key)


def _differ_everywhere(key0: tuple, key1: tuple) -> bool:
    """Whether two groups have a different value in every column."""

    return all(value0 != value1 for value0, value1 in zip(key0, key1))


class PairwiseCompare:
    """
    Compare the rows of a dataframe in pairs, with the rows split into groups twice.

    The rows are first split into *outer* groups by their values in `outer_columns`, and the rows of each outer
    group are then split into *inner* groups by their values in `inner_columns`.
    The rows of two inner groups are compared with the comparator, as long as the two groups have a different value
    in every inner column. Each pair of groups is compared once, in one direction only.

    - `intra_comparisons` compares inner groups of the same outer group.
    - `inter_comparisons` compares inner groups of two outer groups that have a different value in every outer column.

    Users almost always want `pairwise_compare.compare_pairs` instead, which checks how the columns are combined.

    Parameters
    ----------
    df:
        The rows to compare, with the group columns and the feature columns.
    comparator:
        Defines how two groups are compared and how the results are saved. Use a new one for each set of comparisons.
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

        """Check the arguments and prepare the groups (the class docstring describes the parameters)."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("`df` must be a pandas DataFrame.")

        if not isinstance(comparator, Comparator):
            raise TypeError("`comparator` must be an instance of Comparator.")

        outer_columns = as_column_list("outer_columns", outer_columns)
        inner_columns = as_column_list("inner_columns", inner_columns)
        features = as_column_list("features", features)
        drop = as_column_list("drop", drop)

        if df.empty:
            raise ValueError("`df` is empty.")

        missing = (set(outer_columns) | set(inner_columns) | set(features) | set(drop)) - set(df.columns)
        if missing:
            raise ValueError(f"The dataframe does not have the columns {sorted(missing)}.")

        not_numeric = [column for column in features if not pd.api.types.is_numeric_dtype(df[column].dtype)]
        if not_numeric:
            raise TypeError(f"The feature columns {not_numeric} are not numeric.")

        self._warn_about_missing_values(df, outer_columns + inner_columns, features)

        self._comparator = comparator
        self._df = df
        self._features = df[features]
        self._outer_columns = outer_columns
        self._inner_columns = inner_columns

        # the group columns to save with the results, and their positions in the group values
        self._saved_outer = self._saved_columns(outer_columns, drop)
        self._saved_inner = self._saved_columns(inner_columns, drop)

    @staticmethod
    def _saved_columns(group_columns: list[str], drop: list[str]) -> tuple[list[str], list[int]]:
        """The group columns that are not dropped, and their positions among the group columns."""

        positions = [position for position, column in enumerate(group_columns) if column not in drop]
        return [group_columns[position] for position in positions], positions

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

    def _compare(self, rows0: np.ndarray, rows1: np.ndarray) -> None:
        """Compare the rows at two sets of positions."""

        self._comparator(self._features.iloc[rows0], self._features.iloc[rows1])

    def _save(self, saved: tuple[list[str], list[int]], key0: tuple, key1: tuple) -> None:
        """Save the values of the group columns that are not dropped, for the pair of groups just compared."""

        columns, positions = saved

        if columns:
            self._comparator.save_groups(
                columns,
                group0=tuple(key0[position] for position in positions),
                group1=tuple(key1[position] for position in positions),
            )

    def intra_comparisons(self) -> None:
        """Compare the inner groups of each outer group with each other (within the outer groups)."""

        for outer_key, outer_rows in _group_rows(self._df, self._outer_columns).items():
            inner_groups = _group_rows(self._df, self._inner_columns, outer_rows)

            if len(inner_groups) < 2:
                warnings.warn(
                    f"The outer group ({_describe(outer_key)}) has fewer than two inner groups, so nothing is compared in it.",
                    UserWarning,
                )
                continue

            for (inner0, rows0), (inner1, rows1) in combinations(inner_groups.items(), 2):
                if not _differ_everywhere(inner0, inner1):
                    continue

                self._compare(rows0, rows1)
                self._save(self._saved_outer, outer_key, outer_key)
                self._save(self._saved_inner, inner0, inner1)

    def inter_comparisons(self) -> None:
        """Compare the inner groups of two outer groups that have a different value in every outer column."""

        outer_groups = _group_rows(self._df, self._outer_columns)
        inner_groups = {
            outer_key: _group_rows(self._df, self._inner_columns, outer_rows)
            for outer_key, outer_rows in outer_groups.items()
        }

        for outer0, outer1 in combinations(outer_groups, 2):
            if not _differ_everywhere(outer0, outer1):
                continue

            inner_pairs = list(product(inner_groups[outer0].items(), inner_groups[outer1].items()))

            if not inner_pairs:
                warnings.warn(
                    f"The outer groups ({_describe(outer0)}) and ({_describe(outer1)}) have no inner groups to compare.", UserWarning
                )
                continue

            for (inner0, rows0), (inner1, rows1) in inner_pairs:
                if not _differ_everywhere(inner0, inner1):
                    continue

                self._compare(rows0, rows1)
                self._save(self._saved_outer, outer0, outer1)
                self._save(self._saved_inner, inner0, inner1)
