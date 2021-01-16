import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import itertools as it
import json
from network import Network


def read_monk(name, rescale=False):
    """
    Reads the monks dataset
    :param name: name of the dataset (either "monks-1", "monks-2" or "monks-3")
    :param rescale: Whether or not to rescale the targets to [-1, +1]
    :return: monk dataset and labels (as numpy ndarrays)
    """
    # read the dataset
    col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    monk_train = pd.read_csv(f"../datasets/monks/{str(name)}.train", sep=' ', names=col_names)
    monk_train.set_index('Id', inplace=True)
    labels = monk_train.pop('class')

    # 1-hot encoding (and transform dataframe to numpy array)
    monk_train = OneHotEncoder().fit_transform(monk_train).toarray()

    # transform labels from pandas dataframe to numpy ndarray
    labels = labels.to_numpy()[:, np.newaxis]
    if rescale:
        labels[labels == 0] = -1

    # shuffle the whole dataset once
    indexes = list(range(len(monk_train)))
    np.random.shuffle(indexes)
    monk_train = monk_train[indexes]
    labels = labels[indexes]

    return monk_train, labels


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


def list_of_combos(param_dict):
    """
    Takes a dictionary with the combinations of params to use in the grid search and creates a list of dictionaries, one
    for each combination (so it's possible to iterate over this list in the GS, instead of having many nested loops)
    :param param_dict: dict{kind_of_param: tuple of all the values of that param to try in the grid search)
    :return: list of dictionaries{kind_of_param: value of that param}
    """
    combo_list = list(it.product(*(param_dict[k] for k in param_dict.keys())))
    combos = []
    for c in combo_list:
        # check if the current combination is formed of compatible parameters
        # c[0] = units per layer  ;  c[1] = activation functions  -->  their lengths must be equal
        if len(c[0]) == len(c[1]):
            combos.append({'units_per_layer': c[0], 'acts': c[1], 'init_type': c[2], 'lower_lim': c[3],
                           'upper_lim': c[4], 'momentum': c[5], 'batch_size': c[6], 'lr': c[7], 'loss': c[8],
                           'metr': c[9], 'epochs': c[10]})
    return combos


def get_best_models(input_dim, n_models=1):
    with open("../results/results.json", 'r') as f:
        data = json.load(f)
    models, params, errors, std_errors, accuracies, std_accuracies = [], [], [], [], [], []
    for result in data['results']:
        errors.append(result[0])
        std_errors.append(result[1])
        accuracies.append(result[2])
        std_accuracies.append(result[3])

    for i in range(n_models):
        index = np.argmax(accuracies)
        accuracies = np.delete(accuracies, index)
        models.append(Network(input_dim=input_dim, **data['params'][index]))
        params.append(data['params'][index])

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
