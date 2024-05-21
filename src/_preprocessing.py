import numpy as np


class Preprocessing:
    def __init__(self,
                 **kwargs):
        self._func = None
        self._kwargs = kwargs

    def __call__(self, **kws):
        if self._func is None:
            raise NotImplementedError('This preprocessing function has not been implemented yet.')
        return self._func(**self._kwargs, **kws)


class ZscorePreprocessing(Preprocessing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._func = self.process

    @staticmethod
    def calc(arr_like):
        mean = np.mean(arr_like)
        std = np.std(arr_like)
        return (np.asarray(arr_like) - mean) / std

    def process(self, df, axis):
        assert axis in [0, 1]
        return df.apply(lambda x: self.calc(x), axis=axis)


class StandardScalerPreprocessing(Preprocessing):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._func = self.process

    @staticmethod
    def calc(arr_like):
        max_val = np.max(arr_like)
        min_val = np.min(arr_like)
        return (np.asarray(arr_like) - min_val) / max_val

    def process(self, df, axis):
        assert axis in [0, 1]
        return df.apply(lambda x: self.calc(x), axis=axis)


PREPROCESSINGS = {"z_score": ZscorePreprocessing,
                  "standardize": StandardScalerPreprocessing}