import os
import numpy as np
import tqdm
from multiprocessing import Process
from datetime import datetime
from joblib import Parallel, delayed
from utility import plot_curves, sets_from_folds, start_processes_and_wait, list_of_combos
from network import Network


def cross_valid(net, tr_val_x, tr_val_y, loss, metr, lr, lr_decay=None, limit_step=None, opt='gd', momentum=0.,
                epochs=1, batch_size=1, k_folds=5, reg_type='l2', lambd=0, verbose=False):
    # split the dataset into folds
    x_folds = np.array(np.array_split(tr_val_x, k_folds), dtype=object)
    y_folds = np.array(np.array_split(tr_val_y, k_folds), dtype=object)

    # initialize vectors for plots
    tr_error_values, tr_metric_values = np.zeros(epochs), np.zeros(epochs)
    val_error_values, val_metric_values = np.zeros(epochs), np.zeros(epochs)
    val_metric_per_fold, val_error_per_fold = [], []

    # CV cycle
    for i in tqdm.tqdm(range(k_folds), desc='Iterating over folds', disable=True):
        # create validation set and training set using the folds (for one iteration of CV)
        tr_data, tr_targets, val_data, val_targets = sets_from_folds(x_folds, y_folds, val_fold_index=i)

        # compile and fit the model on the current training set and evaluate it on the current validation set
        net.compile(opt=opt, loss=loss, metr=metr, lr=lr, lr_decay=lr_decay, limit_step=limit_step, momentum=momentum,
                    reg_type=reg_type, lambd=lambd)
        tr_history = net.fit(tr_x=tr_data, tr_y=tr_targets, val_x=val_data, val_y=val_targets, epochs=epochs,
                             batch_size=batch_size)

        # metrics for the graph
        # composition of tr_history:
        #   [0] --> training error values for each epoch
        #   [1] --> training metric values for each epoch
        #   [2] --> validation error values for each epoch
        #   [3] --> validation metric values for each epoch
        tr_error_values += tr_history[0]
        tr_metric_values += tr_history[1]
        val_error_values += tr_history[2]
        val_metric_values += tr_history[3]
        val_error_per_fold.append(tr_history[2][-1])
        val_metric_per_fold.append(tr_history[3][-1])

        # reset net's weights for the next iteration of CV
        net = Network(**net.params)

    # average the validation results of every fold
    tr_error_values /= k_folds
    tr_metric_values /= k_folds
    val_error_values /= k_folds
    val_metric_values /= k_folds

    # results
    avg_val_err, std_val_err = np.mean(val_error_per_fold), np.std(val_error_per_fold)
    avg_val_metric, std_val_metric = np.mean(val_metric_per_fold), np.std(val_metric_per_fold)

    # print k-fold metrics
    if verbose:
        print("\nValidation scores per fold:")
        for i in range(k_folds):
            print(f"Fold {i + 1} - Loss: {val_error_per_fold[i]} - Accuracy: {val_metric_per_fold[i]}\n{'-' * 62}")
        print('\nAverage validation scores for all folds:')
        print("Loss: {} - std:(+/- {})\nAccuracy: {} - std:(+/- {})".format(avg_val_err, std_val_err,
                                                                            avg_val_metric, std_val_metric))

    plot_curves(tr_error_values, val_error_values, tr_metric_values, val_metric_values)
    return tr_error_values, tr_metric_values, val_error_values, val_metric_values


def grid_search(dev_set_x, dev_set_y):
    folder_path = "../results/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    start = datetime.now()
    grid_search_params = {
        'units_per_layer': ((10, 2), (15, 2), (10, 10, 2)),
        'acts': (('leaky_relu', 'identity'), ('leaky_relu', 'leaky_relu', 'identity'),),
        'momentum': (0., 0.6, 0.8),
        'batch_size': (30, 15, 'full'),
        'lr': (0.0002, 0.001),
        'init_type': ('uniform',),
        'lower_lim': (0.0001,),
        'upper_lim': (0.001,)
    }
    models = []
    param_combos = list_of_combos(grid_search_params)
    for combo in param_combos:
        models.append(Network(input_dim=len(dev_set_x[0]), units_per_layer=combo['units_per_layer'], acts=combo['acts'],
                              init_type=combo['init_type'], lower_lim=combo['lower_lim'], upper_lim=combo['upper_lim']))

    Parallel(n_jobs=os.cpu_count() - 1, verbose=50)(delayed(cross_valid)(
        net=models[i], tr_val_x=dev_set_x, tr_val_y=dev_set_y, loss='squared', metr='euclidean',
        lr=param_combos[i]['lr'], momentum=param_combos[i]['momentum'], epochs=10,
        batch_size=param_combos[i]['batch_size'], k_folds=5) for i in range(len(param_combos)))

    end = datetime.now()
    print('Time used: ', end - start)
