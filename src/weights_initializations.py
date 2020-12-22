import numpy as np


def weights_inits(**kwargs):
    if 'type' not in kwargs.keys():
        raise AttributeError("'type' must be passed to function weights_inits")
    init_type = kwargs['type']
    inits = {
        'uniform': _uniform_init,
        'random': _rand_init
    }
    return inits[init_type](**kwargs)


def _uniform_init(n_weights, value, **kwargs):
    return [value] * n_weights if n_weights > 1 else value


def _rand_init(n_weights, lower_lim=0., upper_lim=1., **kwargs):
    if lower_lim >= upper_lim:
        raise ValueError(f"lower_lim must be <= than upper_lim")
    if n_weights > 1:
        return np.random.uniform(lower_lim, upper_lim, n_weights)
    else:
        return np.random.randn() % upper_lim
