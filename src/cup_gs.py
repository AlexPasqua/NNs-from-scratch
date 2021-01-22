from utility import read_cup, get_best_models, plot_curves
from model_selection import grid_search

if __name__ == '__main__':
    # read dataset
    devset_x, devset_y, int_ts_x, int_ts_y, ts_data = read_cup(int_ts=True)

    # grid search
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
    # grid_search(data=devset_x, targets=devset_y, ds_name="cup", params=gs_params, coarse=True)
    # _, best_params = get_best_models(dataset="cup", coarse=True, n_models=5)
    # best_params = best_params[0]
    # grid_search(data=devset_x, targets=devset_y, ds_name="cup", params=best_params, coarse=False, n_config=4)

    best_models, best_params = get_best_models(dataset="cup", coarse=False, n_models=10)
    for p in best_params:
        print(p)

    # best_model, best_params = best_models[0], best_params[0]
    # best_params['epochs'] = 400
    # best_model.compile(opt='sgd', **best_params)
    # tr_error_values, tr_metric_values, val_error_values, val_metric_values = best_model.fit(
    #     tr_x=devset_x, tr_y=devset_y, disable_tqdm=False, **best_params)
    #
    # # plot graph
    # plot_curves(
    #     tr_loss=tr_error_values,
    #     val_loss=val_error_values,
    #     tr_acc=tr_metric_values,
    #     val_acc=val_metric_values,
    #     path="gae_cloud.png"
    # )
    #
    # res = best_model.evaluate(inp=int_ts_x, targets=int_ts_y, metr='euclidean', loss='squared', disable_tqdm=False)
    # print(f"Err: {res[0]}\tMetr: {res[1]}")
