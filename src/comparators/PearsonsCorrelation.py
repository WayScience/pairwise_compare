from collections import defaultdict
from typing import Optional, Any

import numpy as np
import pandas as pd
from comparators.Comparator import Comparator


class PearsonsCorrelation(Comparator):
    """Computes and stores group and Pearsons Correlation data between paired groups."""

    def __init__(self, _comparison_name: str = "pearsons_correlation"):

        super().__init__()

        self._comparison_name = _comparison_name

    def _preprocess_data(self):
        self._group0, self._group1 = self._group0.values, self._group1.values

    def save_groups(self, _group_cols: list[str], **_groups: dict[str, tuple[Any, ...]]):
        """Save column values defining comparison groups"""

        comparison_count = self._group0.shape[0] * self._group1.shape[0]

        for idx, col in enumerate(_group_cols):
            for group_name, group in _groups.items():
                self._comparisons[f"{col}__{group_name}"].extend(
                    [group[idx]] * comparison_count
                )

    @property
    def comparisons(self):
        return self._comparisons

    def __call__(self, _group0: pd.DataFrame, _group1: pd.DataFrame):

        self._group0, self._group1 = _group0, _group1
        self._preprocess_data()

        group0_centered = self._group0 - self._group0.mean(axis=1, keepdims=True)
        group1_centered = self._group1 - self._group1.mean(axis=1, keepdims=True)

        dot_products = np.dot(group0_centered, group1_centered.T)

        group0_norms = np.linalg.norm(group0_centered, axis=1, keepdims=True)
        group1_norms = np.linalg.norm(group1_centered, axis=1, keepdims=True)

        norm_products = np.dot(group0_norms, group1_norms.T)

        correlations = (dot_products / norm_products).flatten()

        self._comparisons[self._comparison_name].extend(correlations.tolist())
