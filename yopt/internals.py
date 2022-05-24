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
                    norm = math.sqrt(np.dot(weights.dot(inv_matrix), weights))

                if norm > 1.:
                    weights /= norm

    return done_steps


@jit(nopython=True)
def _md_offline_fit(
    X: ArrayLike,
    y: ArrayLike,
    n_passes: int,
    start_ŋ: float,
    weights: ArrayLike,
    inv_matrix: ArrayLike
):
    y -= .5
    y *= 2

    n, done_steps = X.shape[0], 1

    w = weights.copy()
    g = np.zeros_like(weights)

    for k in range(n_passes):
        for i in range(n):
            features, target = X[i], y[i]
            d = np.dot(w, features) * target

            if d <= 1.:
                if inv_matrix is None:
                    g += target * features
                else:
                    g += inv_matrix @ (target * features)

        g /= n
        curr_ŋ = start_ŋ / math.sqrt(done_steps)
        w += curr_ŋ * g
        g[:] = 0
        done_steps += 1

        if inv_matrix is None:
            norm = np.linalg.norm(w)
        else:
            norm = math.sqrt(np.dot(w.dot(inv_matrix), w))

        if norm > 1.:
            w /= norm

        weights += w

    weights /= n_passes + 1

    return done_steps


