import numpy as np


def weights_inits(init_type, **kwargs):
    # if 'init_type' not in kwargs.keys():
    #     raise AttributeError("'init_type' must be passed to function weights_inits")
    inits = {
        'uniform': _uniform_init,
        'random': _rand_init
    }
    return inits[init_type](**kwargs)


def _uniform_init(n_weights, n_units, init_value, **kwargs):
    return np.full(shape=(n_weights, n_units), fill_value=init_value)
    # return [init_value] * n_weights if n_weights > 1 else init_value


def _rand_init(n_weights, n_units, lower_lim=0., upper_lim=1., **kwargs):
    if lower_lim >= upper_lim:
        raise ValueError(f"lower_lim must be <= than upper_lim")
    res = np.random.uniform(low=lower_lim, high=upper_lim, size=(n_weights, n_units))
    if n_weights == 1:
        return res[0]
    return res
