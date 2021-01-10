import os
from datetime import datetime
from itertools import product
import random as random
import numpy as np


def grid_search(net, params, folds):
    # create directory
    os.makedirs("grid_reports", exist_ok=True)
    timestamp = datetime.today().isoformat().replace(':', '_')
    filename = 'grid_reports/grid_table__'+ timestamp
    # write file
    #with open(filename +'.gsv', 'w') as file:


    # grid_search results
    res_grid = []

    # call cross_valid with hyper-params from get_RandParams


    # append results to res_grid

    # return res_grid

    pass


def get_Params(params):
    """
    Generate a list of dictionaries containing hyper-parameters combination to be used for a coarse grid-search
    """
    res = []
    params = [params]
    for line in params:
        for par in line:
            items = sorted(par.items())

            keys, values = zip(*items)
            for val in product(*values):
                param = dict(zip(keys, val))
                res.append(param)
    return res


def get_RandParams(params):
    """
    Generate a list of dictionaries of random hyper-parameters for a grid search
    """
    params = [params]
    rand_res = []
    for line in params:
        for par in line:
            items = sorted(par.items())
            if not items:
                return
            else:
                pa = {}
                keys, values = zip(*items)
                for k, vl in zip(keys, values):
                    isnumber = all(type(v) in (int, float) for v in vl)
                    if isnumber:
                        pa[k] = [np.random.uniform(min(vl), max(vl))]
                    else:
                        pa[k] = random.choice(vl)

                rand_res.append(pa)
    return rand_res


if __name__ == '__main__':
    hyp_params = [
        {
            'units_per_layer': [(10, 1), (20, 1), (50, 1), (100, 1), (5, 1)],
            'momentum': [0., 0.9],
            'batch_size': [1, 5, 10, 50, 100, 'full'],
            'lr': [0.001, 0.3],
            'learning_rate_init': [0.1, 0.9],
            'lr_decay': ['linear'],
            'limit_step': [200, 5000],
            'acts': [('leaky_relu', 'tanh'),('tanh','tanh'),('leaky_relu','sigmoid'),('relu','tanh'),('relu','sigmoid') ,('sigmoid','sigmoid')],
            'init_type': ['random'],
            'lower_lim': [-0.1, 0.05],
            'upper_lim': [0.1, 0.3]
        }
    ]

    print(get_RandParams(hyp_params))

