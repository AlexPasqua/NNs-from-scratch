import numpy as np


def uniform_init(n_weights, value, **kwargs):
    return [value] * n_weights


def rand_init(lower_lim, upper_lim, n_weights, **kwargs):
    if n_weights > 1:
        return np.random.uniform(lower_lim, upper_lim, n_weights)
    else:
        return np.random.randn() % upper_lim


weights_inits = {
    'uniform': uniform_init,
    'random': rand_init
}
