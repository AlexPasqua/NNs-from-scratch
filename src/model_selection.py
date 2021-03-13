import json
import os
import numpy as np
import tqdm
from datetime import datetime
import warnings
from joblib import Parallel, delayed
from utility import plot_curves, sets_from_folds, list_of_combos, read_monk, read_cup, randomize_params
from network import Network


def cross_valid(net, dataset, loss, metr, lr, path=None, lr_decay=None, limit_step=None, decay_rate=None, decay_steps=None,
                staircase=True, opt='sgd', momentum=0., epochs=1, batch_size=1, k_folds=5, reg_type='l2', lambd=0,
                disable_tqdms=(True, True), plot=True, verbose=False, **kwargs):
    """
    Performs a k-fold cross validation
    :param net: the Network onto which execute the cross validation
    :param dataset: name of the dataset to use
    :param loss: name of the loss function to use ('squared')
    :param metr: name of the metric function to use ('bin_class_acc' ot 'euclidean')
    :param lr: learning rate
    :param path: path where to save the plots (if plot=True)
    :param lr_decay: type of learning rate decay (either None, 'linear' or 'exponential')
    :param limit_step: (int) limit step for the linearly decaying learning rate (in case see functions.py)
    :param decay_rate: (float) decay rate for the exponentially decaying learning rate (in case see functions.py)
    :param decay_steps: (int) like limit_step but for the exponentially decaying learning rate (in case see functions.py)
    :param staircase: (bool) if True, with the exponential decay of the lr, it will decrease in a stair-like fashion
    :param opt: optimizer name ('sgd' for now, open for more)
    :param momentum: (float) momentum coefficient
    :param epochs: (int) number of epochs
    :param batch_size: (int/str) batch size (either 'full' or a number)
    :param k_folds: number of folds for the k-fold cross validation
    :param reg_type: type of regularization ('l1' / 'l2)
    :param lambd: (float) regularization coefficient
    :param disable_tqdms: couple of booleans -> disable progress bars
    :param plot: (bool) if True the plot of the CV will be saved to 'path'
    :return: avg_val_err, std_val_err, avg_val_metric, std_val_metric
    """
    # read the dataset
    if dataset not in ('monks-1.train', 'monks-2.train', 'monks-3.train', 'cup'):
        raise ValueError("Attribute dataset must be in {monks-1.train, monks-2.train, monks-3.train, cup}")
    if dataset == "cup":
        dev_set_x, dev_set_y, _, _, _ = read_cup(int_ts=True)
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
    tr_error_per_fold, tr_metric_per_fold = [], []

    # CV cycle
    for i in tqdm.tqdm(range(k_folds), desc='Iterating over folds', disable=disable_tqdms[0]):
        # create validation set and training set using the folds (for one iteration of CV)
        tr_data, tr_targets, val_data, val_targets = sets_from_folds(x_folds, y_folds, val_fold_index=i)

        # compile and fit the model on the current training set and evaluate it on the current validation set
        net.compile(opt=opt, loss=loss, metr=metr, lr=lr, lr_decay=lr_decay, limit_step=limit_step,
                    decay_rate=decay_rate, decay_steps=decay_steps, staircase=staircase, momentum=momentum,
                    reg_type=reg_type, lambd=lambd)
        warnings.simplefilter("error")
        try:
            tr_history = net.fit(tr_x=tr_data, tr_y=tr_targets, val_x=val_data, val_y=val_targets, epochs=epochs,
                                 batch_size=batch_size, disable_tqdm=disable_tqdms[1])
        except Exception as e:
            print(f"{e.__class__.__name__} occurred. Training suppressed.")
            print(e, '\n')
            return

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
        try:
            tr_error_per_fold.append(tr_history[0][-1])
            tr_metric_per_fold.append(tr_history[1][-1])
            val_error_per_fold.append(tr_history[2][-1])
            val_metric_per_fold.append(tr_history[3][-1])
        except TypeError:
            tr_error_per_fold.append(tr_history[0])
            tr_metric_per_fold.append(tr_history[1])
            val_error_per_fold.append(tr_history[2])
            val_metric_per_fold.append(tr_history[3])

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
    avg_tr_err, std_tr_err = np.mean(tr_error_per_fold), np.std(tr_error_per_fold)
    avg_tr_metr, std_tr_metr = np.mean(tr_metric_per_fold), np.std(tr_metric_per_fold)

    # print k-fold metrics
    if verbose:
        print("\nScores per fold:")
        for i in range(k_folds):
            print(f"Fold {i + 1}:\nVal Loss: {val_error_per_fold[i]} - Val Metric: {val_metric_per_fold[i]}\n"
                  f"Train Loss: {tr_error_per_fold[i]} - Train Metric: {tr_metric_per_fold[i]}\n{'-' * 62}\n")
        print('\nAverage scores for all folds:')
        print("Val Loss: {} - std:(+/- {})\n"
              "Train Loss: {} - std:(+/- {})\n"
              "Val Metric: {} - std:(+/- {})\n"
              "Train Metric: {} - std(+/- {}\n".format(avg_val_err, std_val_err,
                                                       avg_tr_err, std_tr_err,
                                                       avg_val_metric, std_val_metric,
                                                       avg_tr_metr, std_tr_metr))
    if plot:
        ylim, lbltr, lblval = None, None, None
        if "monk" in dataset:
            ylim, lbltr, lblval = (0., 1.1), "Training", "Validation"
        plot_curves(tr_error_values, val_error_values, tr_metric_values, val_metric_values, path, ylim=ylim,
                    lbltr=lbltr, lblval=lblval)
    return avg_val_err, std_val_err, avg_val_metric, std_val_metric


def grid_search(dataset, params, coarse=True, n_config=1):
    """
    Performs a grid search over a set of parameters to find the best combination of hyperparameters
    :param dataset: name of the dataset (monks-1, monks-2, monks-3, cup)
    :param params: dictionary with all the values of the params to try in the grid search
    :param coarse: (bool) if True perform a gird search only on the values of 'params'
    :param n_config: (int) number of config to generate for each param in case of NOT coarse grid search
    """
    models = []
    input_dim = 10 if dataset == "cup" else 17

    # In case generate random combinations
    if not coarse:
        params = randomize_params(params, dataset, n_config)

    # generate list of combinations
    param_combos = list_of_combos(params)
    print(f"Total number of trials: {len(param_combos)}")
    for combo in param_combos:
        models.append(Network(input_dim=input_dim, **combo))

    # perform parallelized grid search
    results = Parallel(n_jobs=os.cpu_count(), verbose=50)(delayed(cross_valid)(
        net=models[i], dataset=dataset, k_folds=5, disable_tqdm=(True, True), plot=False,
        **param_combos[i]) for i in range(len(param_combos)))

    # do not save models with suppressed training
    for r, p in zip(results, param_combos):
        if r is None:
            results.remove(r)
            param_combos.remove(p)

    # write results on file
    folder_path = "../results/"
    file_name = ("coarse_gs_" if coarse else "fine_gs_") + "results_" + dataset + ".json"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    data = {"params": param_combos, "results": results}
    with open(folder_path + file_name, 'w') as f:
        json.dump(data, f, indent='\t')
