from network import Network
from utility import read_monk, plot_curves, get_best_models
from model_selection import grid_search, cross_valid

if __name__ == '__main__':
    # read the dataset
    ds_name = "monks-1.train"
    monk_train, labels = read_monk(name=ds_name, rescale=True)
    x_test, y_test = read_monk(name='monks-1.test', rescale=True)

    model_params = {
        'input_dim': 17,
        'units_per_layer': (4, 1),
        'acts': ('leaky_relu', 'tanh'),
        'init_type': 'uniform',
        'init_value': 0.2,
        'limits': (-0.1, 0.1)
    }
    model = Network(**model_params)

    params = {
        'lr': 0.3,
        'momentum': 0.9,
        # 'lambd': 0.0,
        # 'reg_type': 'l2',
        'lr_decay': 'exponential',
        'decay_rate': 0.95,
        'decay_steps': 250,
        'staircase': False,
        # 'limit_step': 200,
        'loss': 'squared',
        'opt': 'sgd',
        'epochs': 400,
        'batch_size': 'full',
        'metr': 'bin_class_acc',
        'val_split': 0.1
    }

    # # cross validation
    # tr_error_values, tr_metric_values, val_error_values, val_metric_values = cross_valid(
    #     net=model,
    #     dataset="monks-2.train",
    #     k_folds=5,
    #     verbose=False,
    #     disable_tqdms=(True, False),
    #     **params
    # )

    # hold-out validation
    model.compile(**params)
    tr_error_values, tr_metric_values, val_error_values, val_metric_values = model.fit(tr_x=monk_train, tr_y=labels,
                                                                                       disable_tqdm=False, **params)
    pred_test = model.predict(inp=x_test, disable_tqdm=False)
    loss_scores, metr_scores = model.evaluate(net_outputs=pred_test, targets=y_test, metr=params['metr'],
                                              loss=params['loss'], disable_tqdm=False)
    print(f"test_loss:{loss_scores}  -  test_acc: {metr_scores}")

    # grid search
    # gs_params = {'units_per_layer': ((4, 1),),
    #              'acts': (('leaky_relu', 'tanh'), ('leaky_relu', 'leaky_relu', 'tanh')),
    #              'init_type': ('uniform',),
    #              'limits': ((-0.2, 0.2), (-0.001, 0.001)),
    #              'momentum': (0.6, 0.8),
    #              'batch_size': ('full',),
    #              'lr': (0.3, 0.5),
    #              'loss': ('squared',),
    #              'metr': ('bin_class_acc',),
    #              'epochs': (200,)}
    # # grid_search(dataset=ds_name, params=gs_params, coarse=True)
    # best_model, best_params = get_best_models(dataset=ds_name, n_models=1)
    # best_model = best_model[0]
    # best_params = best_params[0]
    # grid_search(dataset=ds_name, params=best_params, coarse=False)
    # best_model, best_params = get_best_models(dataset=ds_name, n_models=1)
    # best_model = best_model[0]
    # best_params = best_params[0]
    # best_model.compile(opt='sgd', **params)
    # tr_error_values, tr_metric_values, val_error_values, val_metric_values = best_model.fit(
    #     tr_x=monk_train, tr_y=labels, disable_tqdm=False, **params)
    #
    # # plot graph
    # plot_curves(
    #     tr_loss=tr_error_values,
    #     val_loss=val_error_values,
    #     tr_acc=tr_metric_values,
    #     val_acc=val_metric_values,
    #     **params
    # )
