import os
from datetime import datetime
from itertools import product
import random as random
import numpy as np
import json
from model_selection import cross_valid
from network.network import Network
from monk_demo import *


def grid_search(params, data, labels, folds, epochs, coarse=True, n_models=1):
    """

    :param params: list of hyper-parameters
    :param data: development set
    :param labels: labels
    :param folds: number of folds of k-fold cv
    :param epochs: epochs
    :param coarse: if False performs a
    :param n_models:
    :return:
    """
    # create directory
    filename = 'grid_test.txt'
    filepath_n = '/Users/gaetanoantonicchio/Desktop/UNIVERSITY OF PISA - DATA SCIENCE/' + filename
    # write file
    with open(filepath_n, 'w') as fp:
        print(f"{datetime.today()}\ngrid_search results:\n", file=fp)
        if coarse:
            grid = get_Params(params=params)
        else:
            grid = get_RandParams(params=params, n_models=n_models)
        res_grid = []
        for i in grid:
            model = Network(input_dim=17, **i)
            r = cross_valid(net=model,
                            tr_val_x=data,
                            tr_val_y=labels,
                            loss='squared',
                            metr='bin_class_acc',
                            k_folds=folds,
                            epochs=epochs,
                            **i)
            res_grid.append([i, r])
            json.dump(i, fp)
            fp.write('\n')
            json.dump(r, fp)
            fp.write('\n')
        return res_grid


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


def get_RandParams(params, n_models):
    """
    Generate a list of dictionaries of random hyper-parameters for a grid search
    """
    params = [params]
    rand_res = []
    for m in range(n_models):
        for line in params:
            for par in params:
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
                                pa[k] = np.random.uniform(min(vl), max(vl))
                            else:
                                pa[k] = random.choice(vl)

                        rand_res.append(pa)
    return rand_res


if __name__ == '__main__':
    hyp_params = [
        {
            'units_per_layer': [(5, 1), (4, 1)],
            'momentum': [0.8, 0.9],
            'batch_size': ['full'],
            'lr': [0.75, 0.8],
            # 'learning_rate_init': [0.8],
            # 'lr_decay': ['linear'],
            # 'limit_step': [200, 5000],
            'acts': [('leaky_relu', 'tanh')],
            'init_type': ['random'],
            'lower_lim': [-0.1],
            'upper_lim': [0.1]
        }
    ]

    monk_train, labels = read_monk(name='monks-1', rescale=True)
    grid_search(params=hyp_params, data=monk_train, labels=labels, folds=10, epochs=4, coarse=False, n_models=1)
    #print(get_RandParams(params=hyp_params, n_models= 4))
    #print(get_Params(params=hyp_params))

