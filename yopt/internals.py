import numpy as np
import math
from numba import jit

from numpy.typing import ArrayLike


@jit(nopython=True)
def _sgd_fit(
    X: ArrayLike,
    y: ArrayLike,
    n_passes: int,
    reg_λ: float,
    start_ŋ: float,
    weights: ArrayLike,
    done_steps: int
):
    n = X.shape[0]
    T = n * n_passes

    y -= .5
    y *= 2

    acc = np.zeros_like(weights)

    for k in range(n_passes):
        indices = np.random.permutation(n)
        for i in indices:
            done_steps += 1
            curr_ŋ = start_ŋ / done_steps
            features, target = X[i], y[i]
            d = np.dot(weights, features) * target
            weights *= 1. - curr_ŋ

            if d <= 1.:
                weights += target * reg_λ * curr_ŋ * features

            acc += weights

    weights = acc / T
    return done_steps


@jit(nopython=True)
def _md_online_fit(
    X: ArrayLike,
    y: ArrayLike,
    n_passes: int,
    start_ŋ: float,
    weights: ArrayLike,
    done_steps: int,
    inv_matrix: ArrayLike,
    md_strategy: int
):
    y -= .5
    y *= 2

    for k in range(n_passes):
        # for features, target in zip(X, y):
        for i in range(X.shape[0]):
            done_steps += 1
            curr_ŋ = start_ŋ if md_strategy == 0 else start_ŋ / done_steps
            features, target = X[i], y[i]
            d = np.dot(weights, features) * target

            if d <= 1.:
                if inv_matrix is None:
                    weights += target * curr_ŋ * features
                    norm = np.linalg.norm(weights)
                else:
                    weights += inv_matrix @ (target * curr_ŋ * features)
                    norm = math.sqrt(weights.T.dot(inv_matrix)*weights.T)

                if norm > 1.:
                    weights /= norm

    return done_steps


