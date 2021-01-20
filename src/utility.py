import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from collections import OrderedDict
import random
import itertools as it
import json
from network import Network


def read_monk(name, rescale=False):
    """
    Reads the monks dataset
    :param name: name of the dataset
    :param rescale: Whether or not to rescale the targets to [-1, +1]
    :return: monk dataset and labels (as numpy ndarrays)
    """
    # read the dataset
    col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    try:
        monk_dataset = pd.read_csv(f"../datasets/monks/{str(name)}", sep=' ', names=col_names)
    except FileNotFoundError:
        monk_dataset = pd.read_csv(f"../../datasets/monks/{str(name)}", sep=' ', names=col_names)
    monk_dataset.set_index('Id', inplace=True)
    labels = monk_dataset.pop('class')

    # 1-hot encoding (and transform dataframe to numpy array)
    monk_dataset = OneHotEncoder().fit_transform(monk_dataset).toarray()

    # transform labels from pandas dataframe to numpy ndarray
    labels = labels.to_numpy()[:, np.newaxis]
    if rescale:
        labels[labels == 0] = -1

    # shuffle the whole dataset once
    indexes = list(range(len(monk_dataset)))
    np.random.shuffle(indexes)
    monk_dataset = monk_dataset[indexes]
    labels = labels[indexes]

    return monk_dataset, labels


def read_cup():
    """
    Reads the CUP training and testing sets
    :return: CUP training data, CUP training targets and CUP test data (as numpy ndarray)
    """
    # read the dataset
    directory = "../datasets/cup/"
    file = "ML-CUP20-TR.csv"
    col_names = ['id', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9', 'a10', 'target_x', 'target_y']
    cup_tr_data = pd.read_csv(directory + file, sep=',', names=col_names, skiprows=range(7), usecols=range(1, 11))
    cup_tr_targets = pd.read_csv(directory + file, sep=',', names=col_names, skiprows=range(7), usecols=range(11, 13))
    file = "ML-CUP20-TS.csv"
    cup_ts_data = pd.read_csv(directory + file, sep=',', names=col_names[: -2], skiprows=range(7))
    cup_tr_data = cup_tr_data.to_numpy()
    cup_tr_targets = cup_tr_targets.to_numpy()
    cup_ts_data = cup_ts_data.to_numpy()

    # shuffle the training dataset once
    indexes = list(range(cup_tr_targets.shape[0]))
    np.random.shuffle(indexes)
    cup_tr_data = cup_tr_data[indexes]
    cup_tr_targets = cup_tr_targets[indexes]

    # standardization / normalization
    # cup_tr_data = StandardScaler().fit_transform(cup_tr_data)
    # cup_tr_targets = MinMaxScaler().fit_transform(cup_tr_targets)

    return cup_tr_data, cup_tr_targets, cup_ts_data


def sets_from_folds(x_folds, y_folds, val_fold_index):
    """
    Takes folds from cross validation and return training and validation sets as a whole (not lists of folds)
    :param x_folds: list of folds containing the data
    :param y_folds: list of folds containing the targets
    :param val_fold_index: index of the fold to use as validation set
    :return: training data set, training targets set, validation data set, validation targets set (as numpy ndarray)
    """
    val_data, val_targets = x_folds[val_fold_index], y_folds[val_fold_index]
    tr_data_folds = np.concatenate((x_folds[: val_fold_index], x_folds[val_fold_index + 1:]))
    tr_targets_folds = np.concatenate((y_folds[: val_fold_index], y_folds[val_fold_index + 1:]))
    # here tr_data_folds & tr_targets_folds are still a "list of folds", we need a single seq as a whole
    tr_data = tr_data_folds[0]
    tr_targets = tr_targets_folds[0]
    for j in range(1, len(tr_data_folds)):
        tr_data = np.concatenate((tr_data, tr_data_folds[j]))
        tr_targets = np.concatenate((tr_targets, tr_targets_folds[j]))
    return tr_data, tr_targets, val_data, val_targets


def randomize_params(base_params, dataset, n_config=2):
    ds = read_cup() if dataset == "cup" else read_monk(dataset)
    fb_dim = len(ds[0])
    n_config -= 1
    rand_params = {}
    for k, v in base_params.items():
        # if the parameter does not have to change
        if k in ('acts', 'init_type', 'decay_rate', 'loss', 'lr_decay', 'metr', 'reg_type', 'staircase',
                 'units_per_layer'):
            rand_params[k] = (v,)
        else:
            rand_params[k] = [v]
            for i in range(n_config):
                # generate n_config random value centered in v
                if k == "batch_size":
                    if v == "full":
                        rand_params[k] = ("full",)
                        continue
                    lower = max(v - 15, 1)
                    upper = min(v + 15, fb_dim)
                    value = random.randint(lower, upper)
                    while value in rand_params[k]:
                        lower = max(v - 15, 1)
                        upper = min(v + 15, fb_dim)
                        value = random.randint(lower, upper)
                    rand_params[k].append(value)
                # elif k == "decay_rate":
                #     if v is not None:
                #         lower = max(0., v - 0.2)
                #         upper = min(1., v + 0.2)
                #         rand_params[k].append(random.uniform(lower, upper))
                #     else:
                #         rand_params[k] = (None,)
                elif k in ("epochs", "limit_step", "decay_steps"):
                    if v is None:
                        rand_params[k] = (None,)
                        continue
                    lower = max(1, v - 100)
                    upper = v + 100
                    rand_params[k].append(random.randint(lower, upper))
                elif k in ("lambd", "lr"):
                    value = max(0., np.random.normal(loc=v, scale=0.0001))
                    while value in rand_params[k]:
                        value = max(0., np.random.normal(loc=v, scale=0.0001))
                    rand_params[k].append(value)
                elif k == "limits":
                    lower, upper = v[0], v[1]
                    lower = np.random.normal(loc=lower, scale=0.1)
                    upper = np.random.normal(loc=upper, scale=0.1)
                    if lower > upper:
                        aux = lower,
                        lower = upper
                        upper = aux
                    rand_params[k].append((lower, upper))
                elif k == "momentum":
                    value = max(0., np.random.normal(loc=v, scale=0.1))
                    while value in rand_params[k] or value > 1.:
                        value = min(1., np.random.normal(loc=v, scale=0.1))
                    rand_params[k].append(value)

    return rand_params


def list_of_combos(param_dict):
    """
    Takes a dictionary with the combinations of params to use in the grid search and creates a list of dictionaries, one
    for each combination (so it's possible to iterate over this list in the GS, instead of having many nested loops)
    :param param_dict: dict{kind_of_param: tuple of all the values of that param to try in the grid search)
    :return: list of dictionaries{kind_of_param: value of that param}
    """
    expected_keys = sorted(['units_per_layer', 'acts', 'init_type', 'limits', 'momentum', 'batch_size', 'lr', 'loss',
                            'metr', 'epochs', 'lr_decay', 'decay_rate', 'decay_steps', 'staircase', 'limit_step',
                            'lambd', 'reg_type'])
    for k in expected_keys:
        if k not in param_dict.keys():
            param_dict[k] = ('l2',) if k == 'reg_type' else ((0,) if k == 'lambd' else (None,))
    param_dict = OrderedDict(sorted(param_dict.items()))
    combo_list = list(it.product(*(param_dict[k] for k in param_dict.keys())))
    combos = []
    for c in combo_list:
        if len(c[expected_keys.index('units_per_layer')]) == len(c[expected_keys.index('acts')]):
            d = {k: c[i] for k, i in zip(expected_keys, range(len(expected_keys)))}
            combos.append(d)
    return combos


def get_best_models(dataset, coarse, n_models=1):
    file_name = ("coarse_gs_" if coarse else "fine_gs_") + "results_" + dataset + ".json"
    with open("../results/" + file_name, 'r') as f:
        data = json.load(f)

    # put the data into apposite lists
    input_dim = 10 if dataset == "cup" else 17
    models, params, errors, std_errors, metrics, std_metrics = [], [], [], [], [], []
    for result in data['results']:
        if result is not None:
            errors.append(round(result[0], 3))
            std_errors.append(round(result[1], 3))
            metrics.append(round(result[2], 3))
            std_metrics.append(round(result[3], 3))

    errors, std_errors = np.array(errors), np.array(std_errors)
    metrics, std_metrics = np.array(metrics), np.array(std_metrics)
    for i in range(n_models):
        # find best metric model and its index
        index_of_best = np.argmin(metrics) if dataset == "cup" else np.argmax(metrics)
        value_of_best = min(metrics) if dataset == "cup" else max(metrics)

        # search elements with the same value
        if len(metrics) > index_of_best + 1:
            indexes = [index_of_best]
            for j in range(index_of_best + 1, len(metrics)):
                if metrics[j] == value_of_best:
                    indexes.append(j)

            std_metr_to_check = std_metrics[indexes]
            value_of_best = min(std_metr_to_check)
            index_of_best = indexes[np.argmin(std_metr_to_check)]
            for j in indexes:
                if std_metrics[j] != value_of_best:
                    indexes.remove(j)

            err_to_check = errors[indexes]
            value_of_best = min(err_to_check)
            index_of_best = indexes[np.argmin(err_to_check)]
            for j in indexes:
                if errors[j] != value_of_best:
                    indexes.remove(j)

            std_err_to_check = std_errors[indexes]
            value_of_best = min(std_err_to_check)
            index_of_best = indexes[np.argmin(std_err_to_check)]
            for j in indexes:
                if std_errors[j] != value_of_best:
                    indexes.remove(j)

        metrics = np.delete(metrics, index_of_best)
        models.append(Network(input_dim=input_dim, **data['params'][index_of_best]))
        params.append(data['params'][index_of_best])

    return models, params


def plot_curves(tr_loss, val_loss, tr_acc, val_acc, lr=None, momentum=None, lambd=None, **kwargs):
    """ Plot the curves of training loss, training metric, validation loss, validation metric """
    figure, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(range(len(tr_loss)), tr_loss, color='b', linestyle='dashed', label='training error')
    ax[0].plot(range(len(val_loss)), val_loss, color='r', label='validation error')
    ax[0].legend(loc='best', prop={'size': 6})
    ax[0].set_xlabel('Epochs', fontweight='bold')
    ax[0].set_ylabel('Loss', fontweight='bold')
    ax[0].set_title(f"eta: {lr} - alpha: {momentum} - lambda: {lambd}")
    ax[0].grid()
    ax[1].plot(range(len(tr_acc)), tr_acc, color='b', linestyle='dashed', label='training accuracy')
    ax[1].plot(range(len(val_acc)), val_acc, color='r', label='validation accuracy')
    ax[1].legend(loc='best', prop={'size': 6})
    ax[1].set_xlabel('Epochs', fontweight='bold')
    ax[1].set_ylabel('Accuracy', fontweight='bold')
    ax[1].set_title(f"eta: {lr} - alpha: {momentum} - lambda: {lambd}")
    # ax[1].set_ylim((0., 1.1))
    ax[1].grid()
    plt.show()
