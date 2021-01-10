import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt


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

    # standardize the targets
    cup_tr_data = StandardScaler().fit_transform(cup_tr_data)
    cup_tr_targets = MinMaxScaler().fit_transform(cup_tr_targets)

    return cup_tr_data, cup_tr_targets, cup_ts_data


def plot_curves(tr_loss, val_loss, tr_acc, val_acc, lr=None, momentum=None, lambd=None, **kwargs):
    # plot learning curve
    figure, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(range(len(tr_loss)), tr_loss, color='b', linestyle='dashed', label='training error')
    ax[0].plot(range(len(val_loss)), val_loss, color='r', label='validation error')
    ax[0].legend(loc='best', prop={'size': 6})
    ax[0].set_xlabel('Epochs', fontweight='bold')
    ax[0].set_ylabel('Loss', fontweight='bold')
    ax[0].grid()
    ax[0].set_title(f"eta: {lr} - alpha: {momentum} - lambda: {lambd}")
    ax[1].plot(range(len(tr_acc)), tr_acc, color='b', linestyle='dashed', label='training accuracy')
    ax[1].plot(range(len(val_acc)), val_acc, color='r', label='validation accuracy')
    ax[1].legend(loc='best', prop={'size': 6})
    ax[1].set_title(f"eta: {lr} - alpha: {momentum} - lambda: {lambd}")
    ax[1].set_xlabel('Epochs', fontweight='bold')
    ax[1].set_ylabel('Accuracy', fontweight='bold')
    ax[1].set_ylim((0., 1.1))
    ax[1].grid()
    plt.show()

