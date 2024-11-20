from abc import ABC, abstractmethod
from collections import defaultdict

import pandas as pd


class Comparator(ABC):
    """
    This is a template class for how comparator modules should be structured to make comparisons.
    When inheriting from this template class ensure this class's constructor is called.
    """

    def __init__(self):
        """
        Store comparisons in self._comparisons to eventually be converted to a pandas datafram, where:
        1. Each key is a column name.
        2. The index at each position in each list corresponds to a row.
        """

        self._comparisons = defaultdict(list)

    @property
    def comparisons(self):
        return self._comparisons

    @abstractmethod
    def save_groups(self, _group_cols: list[str], **_groups: dict[str, pd.DataFrame]):
        """
        Save tracked column values defining comparison groups.
        Tracked columns include columns used to compare groups.
        """
        pass

    @abstractmethod
    def __call__(self, _group0: pd.DataFrame, _group1: pd.DataFrame):
        """
        This function is intended for:
        1. Comparing the two dataframes.
        2. Saving the result in the self._comparisons data structure, which is later converted to a pandas dataframe.
        """
        pass
