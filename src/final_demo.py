from utility import read_cup, get_best_models, similar
from model_selection import grid_search


if __name__ == '__main__':
    # read cup dataset
    devset_x, devset_y, int_ts_x, int_ts_y, ts_data = read_cup(int_ts=True)

    # coarse grid search
    # TODO: metti un po' tutti i parametri che avete provato nelle varie combo
    gs_params = {'units_per_layer': ((20, 20, 2), (8, 8, 8, 8, 8, 2)),
                 'acts': (('leaky_relu', 'identity'), ('tanh', 'identity'),
                          ('leaky_relu', 'leaky_relu', 'identity'),
                          ('tanh', 'tanh', 'identity'), ('tanh', 'tanh', 'tanh', 'tanh', 'tanh', 'identity'),
                          ('leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'leaky_relu', 'identity')),
                 'init_type': ('uniform',),
                 'limits': ((-0.001, 0.001),),
                 'momentum': (0.6,),
                 'batch_size': (1, 100, 'full'),
                 'lr': (0.01, 0.001,),
                 'lr_decay': (None, 'linear', 'exponential'),
                 'limit_step': (500,),
                 'decay_rate': (0.95,),
                 'decay_steps': (500,),
                 'staircase': (True, False),
                 'loss': ('squared',),
                 'metr': ('euclidean',),
                 'epochs': (150, 700)}
    grid_search(data=devset_x, targets=devset_y, ds_name="cup", params=gs_params, coarse=True)

    # select combinations of parameters from the coarse grid search's result
    _, best_params = get_best_models(dataset="cup", coarse=True, n_models=100)
    selected_params = []
    found_similar = False
    for combo in best_params:
        if len(selected_params) >= 3:
            break
        for saved in selected_params:
            if similar(combo, saved):
                found_similar = True
                break
        if found_similar:
            found_similar = False
            continue
        else:
            selected_params.append(combo)

    # for each one of the best combos of params (that differ substantially from one another), do a rand grid search
    ens_models, ens_params = [], []
    for combo in selected_params:
        grid_search(data=devset_x, targets=devset_y, ds_name="cup", params=combo, coarse=False, n_config=4)
        best_models, best_params = get_best_models(dataset="cup", coarse=False, n_models=1)
        ens_models.append(best_models[0])
        ens_params.append(best_params[0])

    # create an ensembler of models
    # TODO: add ensembler section (ensembler.py present in apposite branch --> merge when you finished all gs)
