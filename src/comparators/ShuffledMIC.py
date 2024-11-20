import numpy as np
from comparators.MIC import MIC


class ShuffledMIC(MIC):
    """
    Shuffle each MIC group data before comparing both groups.
    """

    def _preprocess_data(self):
        self._group0, self._group1 = self._group0.values, self._group1.values
        np.random.shuffle(self._group0)
        np.random.shuffle(self._group1)
