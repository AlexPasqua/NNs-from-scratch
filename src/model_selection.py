import json
import os
import numpy as np
import tqdm
from datetime import datetime
from joblib import Parallel, delayed
from utility import plot_curves, sets_from_folds, list_of_combos, read_monk, read_cup
from network import Network


def cross_valid(net, dataset, loss, metr, lr, lr_decay=None, limit_step=None, decay_rate=None, decay_steps=None,
                staircase=True, opt='sgd', momentum=0., epochs=1, batch_size=1, k_folds=5, reg_type='l2', lambd=0,
                disable_tqdms=(True, True), verbose=False, **kwargs):
    # read the dataset
    if dataset not in ('monks-1.train', 'monks-2.train', 'monks-3.train', 'cup'):
        raise ValueError("Attribute dataset must be in {monks-1.train, monks-2.train, monks-3.train, cup}")
    if dataset == "cup":
        dev_set_x, dev_set_y, _ = read_cup()
    else:
        rescale = True if net.params['acts'][-1] in ('tanh',) else False
        dev_set_x, dev_set_y = read_monk(name=dataset, rescale=rescale)

    # split the dataset into folds
    x_folds = np.array(np.array_split(dev_set_x, k_folds), dtype=object)
    y_folds = np.array(np.array_split(dev_set_y, k_folds), dtype=object)

    # initialize vectors for plots
    tr_error_values, tr_metric_values = np.zeros(epochs), np.zeros(epochs)
    val_error_values, val_metric_values = np.zeros(epochs), np.zeros(epochs)
    val_metric_per_fold, val_error_per_fold = [], []

    # CV cycle
    for i in tqdm.tqdm(range(k_folds), desc='Iterating over folds', disable=disable_tqdms[0]):
        # create validation set and training set using the folds (for one iteration of CV)
        tr_data, tr_targets, val_data, val_targets = sets_from_folds(x_folds, y_folds, val_fold_index=i)

        # compile and fit the model on the current training set and evaluate it on the current validation set
        net.compile(opt=opt, loss=loss, metr=metr, lr=lr, lr_decay=lr_decay, limit_step=limit_step,
                    decay_rate=decay_rate, decay_steps=decay_steps, staircase=staircase, momentum=momentum,
                    reg_type=reg_type, lambd=lambd)
        tr_history = net.fit(tr_x=tr_data, tr_y=tr_targets, val_x=val_data, val_y=val_targets, epochs=epochs,
                             batch_size=batch_size, disable_tqdm=disable_tqdms[1])

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

    plot_curves(tr_error_values, val_error_values, tr_metric_values, val_metric_values, lr, momentum)
    return avg_val_err, std_val_err, avg_val_metric, std_val_metric


def get_coarse_gs_params():
    """
    :return: dictionary of all the parameters to try in a grid search
    """
    return {'units_per_layer': ((4, 1),),
            'acts': (('leaky_relu', 'tanh'), ('leaky_relu', 'leaky_relu', 'tanh')),
            'init_type': ('uniform',),
            'limits': ((-0.2, 0.2), (-0.001, 0.001)),
            'momentum': (0.0, 0.6, 0.8),
            'batch_size': ('full',),
            'lr': (0.3, 0.5),
            'loss': ('squared',),
            'metric': ('bin_class_acc',),
            'epochs': (200,)}


def grid_search(dataset):
    """
    Performs a grid search over a set of parameters to find the best combination of hyperparameters
    :param dataset: name of the dataset (monks-1, monks-2, monks-3, cup)
    """
    models = []
    input_dim = 10 if dataset == "cup" else 17
    grid_search_params = get_coarse_gs_params()
    param_combos = list_of_combos(grid_search_params)
    print(f"Total number of trials: {len(param_combos)}")
    for combo in param_combos:
        models.append(Network(input_dim=input_dim, **combo))

    results = Parallel(n_jobs=os.cpu_count(), verbose=50)(delayed(cross_valid)(
        net=models[i], dataset=dataset, k_folds=5, disable_tqdm=(True, True),
        **param_combos[i]) for i in range(len(param_combos)))

    # write results on file
    folder_path = "../results/"
    file_name = "results.json"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    data = {"params": param_combos, "results": results}
    with open(folder_path + file_name, 'w') as f:
        json.dump(data, f, indent='\t')
