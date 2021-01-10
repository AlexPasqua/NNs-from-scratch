import os
from datetime import datetime
from itertools import product


def grid_search(net, params, folds):
    # create directory
    os.makedirs("grid_reports", exist_ok=True)
    timestamp = datetime.today().isoformat().replace(':', '_')

    filename = "grid_reports/" + net.__class__.__name__ + "_" + timestamp

    with open(filename + ".gsv", 'w', buffering=1):
        pass


def get_Params(params):
    """
    Generate a list of dictionaries containing hyper-parameters combination to be used for a coarse grid-search
    :param params: list of dictionaries of hyper-parameter values
    """
    res = []
    params = [params]
    for line in params:
        for p in line:
            items = sorted(p.items())

            keys, values = zip(*items)
            for v in product(*values):
                par = dict(zip(keys, v))
                res.append(par)
    return res


def get_randParams():
    """
    Generate random hyper-parameters for a grid search
    """
    pass


if __name__ == '__main__':

    hyp_params = [
        {
            'units_per_layer': [(10, 1), (20, 1), (50, 1), (100, 1), (5, 1)],
            'momentum': [0., 0.9],
            'batch_size': [1, 5, 10, 50, 100, 'full'],
            'lr': [0.001, 0.3],
            'learning_rate_init': [0.1, 2],
            'lr_decay': ['linear'],
            'limit_step': [200, 300, 400, 500, 1000, 2000],
            'acts': ['leaky_relu', 'relu', 'tanh', 'sigmoid'],
            'init_type': ['random'],
            'lower_lim': [-0.1, 0.05],
            'upper_lim': [0.1, 0.3]
        }
    ]

    print(get_Params(hyp_params))
