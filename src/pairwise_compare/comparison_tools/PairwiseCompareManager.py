"""The entry points: compare the rows of a dataframe in pairs, by the columns that define the groups."""

import warnings
from collections.abc import Sequence
from typing import Optional, Union

import pandas as pd

from pairwise_compare.comparators.Comparator import Comparator
from pairwise_compare.comparators.PearsonsCorrelation import PearsonsCorrelation
from pairwise_compare.comparison_tools.PairwiseCompare import PairwiseCompare, as_column_list


class PairwiseCompareManager:
    """
    Compare the rows of a dataframe in pairs, as `compare_pairs` does, and keep the results.

    The comparisons are made when the object is created. Calling the object returns the results as a dataframe.
    See `compare_pairs` for the parameters.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        different: list[str],
        features: list[str],
        same: Optional[list[str]] = None,
        drop: Optional[list[str]] = None,
        comparator: Optional[Union[Comparator, Sequence[Comparator]]] = None,
    ):
        """Check the arguments and compare the rows (`compare_pairs` describes the parameters)."""

        different = as_column_list("different", different)
        same = as_column_list("same", same)
        drop = as_column_list("drop", drop)
        features = as_column_list("features", features)

        if not different:
            raise ValueError("`different` must have at least one column.")

        if not same and len(different) < 2:
            raise ValueError(
                "Without `same` columns, `different` must have at least two columns. "
                "Give at least one `same` column or a second `different` column."
            )

        if set(same) & set(different):
            raise ValueError(
                f"`same` and `different` cannot have columns in common: {sorted(set(same) & set(different))}."
            )

        if not features:
            raise ValueError("`features` must have at least one column to compare.")

        not_group_columns = set(drop) - set(same) - set(different)
        if not_group_columns:
            raise ValueError(
                f"`drop` can only have columns of `same` or `different`, not {sorted(not_group_columns)}."
            )

        comparator = PearsonsCorrelation() if comparator is None else comparator

        if same:
            # compare rows of different values in `different`, within each group of `same`
            comparison = PairwiseCompare(
                df, comparator, outer_columns=same, inner_columns=different, features=features, drop=drop
            )
            self._results = comparison.intra_comparisons()

        else:
            # nothing to keep the same: the groups of the first column are compared, and then the groups of the rest
            comparison = PairwiseCompare(
                df, comparator, outer_columns=different[:1], inner_columns=different[1:], features=features, drop=drop
            )
            self._results = comparison.inter_comparisons()

        if self._results.empty:
            warnings.warn("No rows were compared: no two groups have different values in all the `different` columns.", UserWarning)

    def __call__(self) -> pd.DataFrame:
        """The results of the comparisons, with one row per pair of compared rows."""

        return self._results


def compare_pairs(
    df: pd.DataFrame,
    different: list[str],
    features: list[str],
    same: Optional[list[str]] = None,
    drop: Optional[list[str]] = None,
    comparator: Optional[Union[Comparator, Sequence[Comparator]]] = None,
) -> pd.DataFrame:
    """
    Compare the rows of a dataframe in pairs, such as the wells of different cell lines, across their features.

    Two rows are compared when they have a different value in every `different` column and, if `same` is given,
    the same value in every `same` column. Each pair of rows is compared once (rows are not compared with themselves,
    and a pair is not repeated in the other direction).

    Parameters
    ----------
    df:
        The rows to compare, as a tidy dataframe: one row per sample (for example, one well), with metadata columns
        that define the groups (used in `different`, `same` and `drop`) and numeric feature columns (`features`).
    different:
        The columns whose values must all differ between two compared rows (for example, the well and the cell line).
        It needs at least two columns when `same` is not given.
    features:
        The numeric feature columns to compare.
    same:
        The columns whose values must match between two compared rows (for example, the plate). Optional.
    drop:
        Columns of `same` or `different` to leave out of the results. They still decide which rows are compared.
    comparator:
        How two rows are compared. The default is the Pearson correlation (`PearsonsCorrelation`).
        Give a list, such as `[PearsonsCorrelation(), CosineSimilarity()]`, to get one column for each comparator
        from the same pairs of rows (the comparators need different names).
        A comparator keeps no results, so the same one can be used for any number of calls.

    Returns
    -------
    pd.DataFrame
        One row per pair of compared rows, with the comparison (the column `pearsons_correlation` by default, and
        one column for each comparator of a list) and the values of the columns of the two rows of the pair, in the columns `<column>__group0` and
        `<column>__group1` (for the `same` columns both have the same value).
        The first row of a pair is the one that comes first by its values in the group columns.
        The pairs are sorted by the `same` columns (without `same` columns, by the first `different` column of the
        first row and then of the second row), then by the other `different` columns of the first row and then of the
        second row, and then by the position of the rows in `df`.

    Notes
    -----
    Rows with a missing value in a group column (`same` or `different`) are left out, with a warning.
    Rows with a missing value in a feature column are kept, and their comparisons are missing.

    The results table needs memory for every pair of compared rows, and finding the pairs inside one group of `same`
    values needs memory for every combination of two rows of the group. In a test with one group they peaked at about
    80 bytes for every combination (6,000 rows: 2.9 GB), so split very large groups with another `same` column.

    Examples
    --------
    >>> compare_pairs(df, different=["well", "cell_line"], features=feature_columns, same=["plate"])  # doctest: +SKIP
    """

    return PairwiseCompareManager(
        df, different=different, features=features, same=same, drop=drop, comparator=comparator
    )()
