import numpy as np
from MIC import MIC


class ShuffledMIC(MIC):

    def _preprocess_data(self):
        self._group0, self._group1 = self._group0.iloc[0].values, self._group1.iloc[0].values
        np.random.shuffle(self._group0)
        np.random.shuffle(self._group1)
