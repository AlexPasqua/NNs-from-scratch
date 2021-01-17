from network import Network
from utility import read_monk, plot_curves, get_best_models
from model_selection import grid_search, cross_valid
from functions import losses


if __name__ == '__main__':
    # read the dataset
    monk_train, labels = read_monk(name='monks-1.train', rescale=True)

    model_params = {
        'input_dim': 17,
        'units_per_layer': (15, 1),
        'acts': ('relu', 'tanh'),
        'init_type': 'uniform',
        'init_value': 0.2,
        'limits': (-0.1, 0.1)
    }
    model = Network(**model_params)

    params = {
        'lr': 0.6,
        'momentum': 0.9,
        # 'lambd': 0.0,
        # 'reg_type': 'l2',
        # 'lr_decay': 'exponential',
        # 'decay_rate': 0.95,
        # 'decay_steps': 500,
        # 'staircase': False,
        # 'limit_step': 200,
        'loss': 'squared',
        'opt': 'sgd',
        'epochs': 100,
        'batch_size': 'full',
        'metr': 'bin_class_acc'
    }

    # cross validation
    #tr_error_values, tr_metric_values, val_error_values, val_metric_values = cross_valid(
    #    net=model,
    #    dataset="monks-2",
    #    k_folds=5,
    #    verbose=False,
    #    disable_tqdms=(True, False),
    #    **params
    #)

    # # hold-out validation
    # model.compile(opt='sgd', loss='squared', metr='bin_class_acc', lr=0.2, momentum=0.6)
    # tr_error_values, tr_metric_values, val_error_values, val_metric_values = model.fit(
    #     tr_x=monk_train,
    #     tr_y=labels,
    #     epochs=100,
    #     val_split=0.1,
    #     batch_size='full',
    #     disable_tqdm=False
    # )

    # # grid search
    # grid_search(dataset="monks-1")
    # best_model, params = get_best_models(n_models=1, input_dim=len(monk_train[0]))
    # best_model = best_model[0]
    # params = params[0]
    # best_model.print_topology()
    # best_model.compile(opt='sgd', **params)
    # tr_error_values, tr_metric_values, val_error_values, val_metric_values = best_model.fit(
    #     tr_x=monk_train, tr_y=labels, disable_tqdm=False, **params)

    # # plot graph
    # plot_curves(
    #     tr_loss=tr_error_values,
    #     val_loss=val_error_values,
    #     tr_acc=tr_metric_values,
    #     val_acc=val_metric_values,
    #     **params
    # )

x_test, y_test = read_monk(name='monks-1.test', rescale=True)

model.compile(**params)
model.fit(**params, tr_x=monk_train, tr_y=labels, display_scores=False)
pred_test = model.forward(inp=x_test)
results = model.evaluate(predicted=pred_test, labels=y_test, metr='bin_class_acc', loss='squared')
print(results)


