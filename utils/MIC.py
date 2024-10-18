from collections import defaultdict
from typing import Optional

import pandas as pd
from Comparator import Comparator
from minepy import MINE


class MIC(Comparator):
    """Computes and stores group and Maximal Information Coefficient (MIC) data between paired groups"""

    def __init__(self, _comparison_name: str = "mic_e", _mine_params: Optional[dict[str, str]] = None):
        self._comparisons = defaultdict(list)

        self._comparison_name = _comparison_name

        if _mine_params:
            self._mine_params = _mine_params

        else:
            self._mine_params = {
                "est": "mic_e"
            }

    def save_groups(self, _group_cols: list[str], **_groups: dict[str, pd.DataFrame]):
        """Save column values defining comparison groups"""

        for idx, col in enumerate(_group_cols):
            for group_name, group in _groups.items():
                if len(_group_cols) > 1:
                    self._comparisons[f"{col}__{group_name}"].append(group[idx])
                else:
                    self._comparisons[f"{col}__{group_name}"].append(group)

    def _preprocess_data(self):
        self._group0, self._group1 = self._group0.iloc[0].values, self._group1.iloc[0].values

    @property
    def comparisons(self):
        return self._comparisons

    def __call__(self, _group0: pd.DataFrame, _group1: pd.DataFrame):
        """Compute Comparisons between groups"""

        self._group0, self._group1 = _group0, _group1
        self._preprocess_data()

        mine = MINE(**self._mine_params)

        mine.compute_score(self._group0, self._group1)
        self._comparisons[self._comparison_name].append(mine.mic())
