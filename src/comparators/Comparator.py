from abc import ABC, abstractmethod


class Comparator(ABC):

    @abstractmethod
    def save_groups(self, _cols, _drop_cols, _groups):
        pass

    @abstractmethod
    def _preprocess_data(self):
        pass

    @abstractmethod
    def __call__(self, _group0, _group1):
        pass
