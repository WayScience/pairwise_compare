from typing import Optional

import pandas as pd
from Comparator import Comparator
from PairwiseCompare import PairwiseCompare


class PairwiseCompareManager:

    def __init__(
        self,
        _df: pd.DataFrame,
        _different_columns: list[str],
        _feat_cols: list[str],
        _comparator: Comparator,
        _same_columns: Optional[list[str]] = None,
        _drop_cols: Optional[list[str]] = None
    ):
        """
        _same_columns: The values of these columns must all be the same when comparing any two instances (rows).
        _different_columns: The values of these columns must all be different when comparing any two instances (rows).
        """

        if (not _same_columns and len(_different_columns) == 1):
            raise ValueError("Must specify at least two different columns or at least one different column and at least one same column.")

        if len(_feat_cols) == 0:
            raise ValueError("You must specify at least one feature to compare between samples")

        self._comparator = _comparator

        if len(_different_columns) >= 2 and not _same_columns:
            pairwise_compare_obj = PairwiseCompare(
                _df=_df,
                _comparator=_comparator,
                _antehoc_group_cols=_different_columns[:1],
                _posthoc_group_cols=_different_columns[1:],
                _feat_cols=_feat_cols,
                _drop_cols=_drop_cols
            )
            pairwise_compare_obj.inter_comparisons()

        elif len(_different_columns) >= 1 and len(_same_columns) >= 1:
            pairwise_compare_obj = PairwiseCompare(
                _df=_df,
                _comparator=_comparator,
                _antehoc_group_cols=_same_columns,
                _posthoc_group_cols=_different_columns,
                _feat_cols=_feat_cols,
                _drop_cols=_drop_cols
            )
            pairwise_compare_obj.intra_comparisons()

    def __call__(self):
        return pd.DataFrame(self._comparator.comparisons)
