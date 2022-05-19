import numpy as np

from sklearn.multiclass import OneVsOneClassifier

import enum
from typing import Any, Dict, Union
from numpy.typing import ArrayLike

from .internals import _sgd_fit, _md_online_fit


class MirrorDescentStepStrategy(enum.IntEnum):
    CONST = 0
    DECREASING = 1


class SVMBinaryClassifier:
    """
    Представляет классификатор на основе метода опорных векторов для
    решения задачи бинарной классификации.
    """
    def __init__(self, reg_λ: float=1., start_ŋ: float=1e-1, n_passes: int=10,
        method: str='sgd',
        mode: str='offline',
        md_inv_matrix: Union[ArrayLike, None]=None,
        md_strategy: MirrorDescentStepStrategy=MirrorDescentStepStrategy.CONST
    ):
        assert reg_λ > 0
        assert start_ŋ > 0
        assert n_passes > 0

        self._weights = None
        self.reg_λ = reg_λ
        self.start_ŋ = start_ŋ
        self.n_passes = n_passes
        self.method = method
        self.mode = mode
        self.md_inv_matrix = md_inv_matrix
        self.md_strategy = md_strategy
        

    def get_params(self, deep: bool=True) -> Dict[str, Any]:
        return dict(
            reg_λ=self.reg_λ,
            start_ŋ=self.start_ŋ,
            n_passes=self.n_passes,
            method=self.method,
            mode=self.mode,
            md_inv_matrix=self.md_inv_matrix,
            md_strategy=self.md_strategy
        )


    def set_params(self, **parameters) -> 'SVMBinaryClassifier':
        for param, value in parameters.items():
            setattr(self, param, value)
        return self


    def fit(self, X: ArrayLike, y: ArrayLike) -> 'SVMBinaryClassifier':
        assert X.shape[0] == len(y)

        X = np.asarray(X, dtype=np.float64).copy()
        y = np.asarray(y, dtype=np.float64).copy()

        if self._weights is None or self.mode == 'offline':
            self._weights = np.zeros(X.shape[1], dtype=np.float64, order='C')
            self._done_steps = 0

        if self.method == 'sgd':
            self._done_steps = _sgd_fit(X, y, self.n_passes, self.reg_λ, self.start_ŋ, self._weights, self._done_steps)
        elif self.method == 'md':
            self._done_steps = _md_online_fit(X, y,
                self.n_passes,
                self.start_ŋ,
                self._weights,
                self._done_steps,
                self.md_inv_matrix,
                self.md_strategy
            )

        return self


    def predict(self, X: ArrayLike) -> ArrayLike:
        return np.float32(np.inner(X, self._weights) > 0)

    
    def decision_function(self, X: ArrayLike) -> ArrayLike:
        return np.inner(X, self._weights)



class SVMClassifier:
    """
    Представляет классификатор на основе метода опорных векторов для
    решения задачи многоклассовой классификации.
    """
    def __init__(self, reg_λ: float=1., start_ŋ: float=1e-1, n_passes: int=10,
        method: str='sgd',
        mode: str='offline',
        md_inv_matrix: Union[ArrayLike, None]=None,
        md_strategy: MirrorDescentStepStrategy=MirrorDescentStepStrategy.CONST
    ):
        self.svm = SVMBinaryClassifier(reg_λ, start_ŋ, n_passes, method, mode, md_inv_matrix, md_strategy)
        self.inner_clf = OneVsOneClassifier(self.svm)


    def get_params(self, deep: bool=True) -> Dict[str, Any]:
        return self.svm.get_params(deep)

    
    def set_params(self, **parameters) -> 'SVMClassifier':
        self.svm.set_params(**parameters)
        return self


    def fit(self, X: ArrayLike, y: ArrayLike) -> 'SVMClassifier':
        self.inner_clf.fit(X, y)
        return self


    def predict(self, X: ArrayLike) -> ArrayLike:
        return self.inner_clf.predict(X) 