from collections import defaultdict
from typing import Optional

import pandas as pd
from comparators.Comparator import Comparator
from minepy import MINE


class MIC(Comparator):
    """Computes and stores group and Maximal Information Coefficient (MIC) data between paired groups."""

    def __init__(
        self,
        _comparison_name: str = "mic_e",
        _mine_params: Optional[dict[str, str]] = None,
    ):

        super().__init__()

        self._comparison_name = _comparison_name

        # MIC params
        if _mine_params:
            self._mine_params = _mine_params

        else:
            self._mine_params = {"est": "mic_e"}

    def save_groups(self, _group_cols: list[str], **_groups: dict[str, pd.DataFrame]):

        comparison_count = self._group0.shape[0] * self._group1.shape[0]

        for idx, col in enumerate(_group_cols):
            for group_name, group in _groups.items():

                if len(_group_cols) > 1:
                    self._comparisons[f"{col}__{group_name}"].extend(
                        [group[idx]] * comparison_count
                    )
                else:
                    self._comparisons[f"{col}__{group_name}"].extend(
                        [group] * comparison_count
                    )

    def _preprocess_data(self):
        self._group0, self._group1 = self._group0.values, self._group1.values

    def __call__(self, _group0: pd.DataFrame, _group1: pd.DataFrame):
        """Compute Comparisons between groups."""

        self._group0, self._group1 = _group0, _group1
        self._preprocess_data()

        for group0_row in range(self._group0.shape[0]):

            for group1_row in range(self._group1.shape[0]):

                mine = MINE(**self._mine_params)
                mine.compute_score(
                    self._group0[group0_row, :], self._group1[group1_row, :]
                )
                self._comparisons[self._comparison_name].append(mine.mic())
                del mine
