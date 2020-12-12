import numpy as np


def weights_inits(**kwargs):
    if 'type' not in kwargs.keys():
        raise AttributeError("'type' must be passed to function weights_inits")
    init_type = kwargs['type']
    inits = {
        'uniform': uniform_init,
        'random': rand_init
    }
    return inits[init_type](**kwargs)


def uniform_init(n_weights, value, **kwargs):
    return [value] * n_weights if n_weights > 1 else value


def rand_init(lower_lim, upper_lim, n_weights, **kwargs):
    if n_weights > 1:
        return np.random.uniform(lower_lim, upper_lim, n_weights)
    else:
        return np.random.randn() % upper_lim

