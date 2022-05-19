import numpy as np
from numpy.typing import ArrayLike
from numba import jit


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
            curr_ŋ = start_ŋ / (done_steps + 1.)
            features, target = X[i], y[i]
            d = np.dot(weights, features) * target
            weights *= 1. - curr_ŋ

            if d <= 1.:
                weights += target * reg_λ * curr_ŋ * features

            acc += weights

    weights = acc / T
    return done_steps