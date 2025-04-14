from typing import Optional

import pandas as pd
from comparators.Comparator import Comparator

from comparison_tools.PairwiseCompare import PairwiseCompare


class PairwiseCompareManager:
    """
    Allows the user to make comparisons between groups in a pandas dataframe as specified by the columns defining the comparison groups.
    See the function parameter descriptions of the constructor, and the README for more info.
    """

    def __init__(
        self,
        _df: pd.DataFrame,
        _different_columns: list[str],
        _feat_cols: list[str],
        _comparator: Comparator,
        _same_columns: Optional[list[str]] = None,
        _drop_cols: Optional[list[str]] = None,
    ):
        """
        _df: Data to compare, which must contain the specified columns passed as arguments. All rows in this dataframe will be compared at least once.
        _different_columns: The values of these columns must all be different when comparing any two groups.
        _feat_cols: Feature columns to be used in comparisons from _df.
        _comparator: The module to use for comparing data.
        _same_columns: The values of these columns will all be the same when comparing any two groups.
        _drop_cols: Columns in the final output datarame, which are excluded.
        """

        # Ensure no duplicate columns
        _feat_cols = list(set(_feat_cols))

        if _same_columns:
            _same_columns = sorted(set(_same_columns))

        if _different_columns:
            _different_columns = sorted(set(_different_columns))

        if _drop_cols:
            _drop_cols = list(set(_drop_cols))

        if _same_columns and _different_columns:
            if len(set(_same_columns) & set(_different_columns)) > 0:
                raise ValueError(
                    "_same_columns and _different_columns cannot have any columns in common."
                )

        if not _same_columns and len(_different_columns) == 1:
            raise ValueError(
                "Must specify at least two different columns or at least one different column and at least one same column."
            )

        if len(_feat_cols) == 0:
            raise ValueError(
                "You must specify at least one feature to compare between samples"
            )

        self._comparator = _comparator

        if len(_different_columns) >= 2 and not _same_columns:
            pairwise_compare_obj = PairwiseCompare(
                _df=_df,
                _comparator=_comparator,
                _antehoc_group_cols=_different_columns[:1],
                _posthoc_group_cols=_different_columns[1:],
                _feat_cols=_feat_cols,
                _drop_cols=_drop_cols,
            )

            pairwise_compare_obj.inter_comparisons()

        elif len(_different_columns) >= 1 and len(_same_columns) >= 1:
            pairwise_compare_obj = PairwiseCompare(
                _df=_df,
                _comparator=_comparator,
                _antehoc_group_cols=_same_columns,
                _posthoc_group_cols=_different_columns,
                _feat_cols=_feat_cols,
                _drop_cols=_drop_cols,
            )

            pairwise_compare_obj.intra_comparisons()

    def __call__(self):
        """Final comparisons dataframe."""
        return pd.DataFrame(self._comparator.comparisons)
