import numpy as np


def weights_inits(init_type, **kwargs):
    inits = {
        'fixed': _fixed_init,
        'uniform': _rand_init
    }
    return inits[init_type](**kwargs)


def _fixed_init(n_weights, n_units, init_value, **kwargs):
    if n_weights == 1:
        return np.full(shape=n_units, fill_value=init_value)
    return np.full(shape=(n_weights, n_units), fill_value=init_value)


def _rand_init(n_weights, n_units, limits=(-0.1, 0.1), **kwargs):
    lower_lim, upper_lim = limits[0], limits[1]
    if lower_lim >= upper_lim:
        raise ValueError(f"lower_lim must be <= than upper_lim")
    res = np.random.uniform(low=lower_lim, high=upper_lim, size=(n_weights, n_units))
    if n_weights == 1:
        return res[0]
    return res
