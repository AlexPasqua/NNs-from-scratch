from utility import read_cup, get_best_models
from model_selection import grid_search

if __name__ == '__main__':
    # read dataset
    devset_x, devset_y, int_ts_x, int_ts_y, ts_data = read_cup(int_ts=True)

    # grid search parameters
    gs_params = {'units_per_layer': ((20, 2), (20, 20, 2), (20, 20, 10, 2)),
                 'acts': (('leaky_relu', 'identity'), ('tanh', 'identity'),
                          ('leaky_relu', 'leaky_relu', 'identity'), ('tanh', 'tanh', 'identity'),
                          ('leaky_relu', 'leaky_relu', 'leaky_relu', 'identity'),
                          ('tanh', 'tanh', 'tanh', 'identity')),
                 'init_type': ('uniform',),
                 'limits': ((-0.001, 0.001),),
                 'momentum': (0.3, 0.5, 0.8),
                 'batch_size': (100,),
                 'lr': (0.01, 0.001, 0.0001),
                 'lr_decay': (None, 'linear', 'exponential'),
                 'limit_step': (400,),
                 'decay_rate': (0.95,),
                 'decay_steps': (400,),
                 'lambd': (0, 0.001, 0.0001, 0.00001),
                 'reg_type': ('l2',),
                 'staircase': (True, False),
                 'loss': ('squared',),
                 'metr': ('euclidean',),
                 'epochs': (150, 400, 800)}

    # coarse to fine grid search. Results are saved on file
    grid_search(dataset="cup", params=gs_params, coarse=True)
    # _, best_params = get_best_models(dataset="cup", coarse=True, n_models=5)
    # best_params = best_params[0]
    # grid_search(dataset="cup", params=best_params, coarse=False, n_config=4)
    # best_models, best_params = get_best_models(dataset="cup", coarse=False, n_models=10)
    # for p in best_params:
    #     print(p)
