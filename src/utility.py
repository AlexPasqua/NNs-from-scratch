import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


def read_monk(filename):
    # read the dataset
    col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    monk_train = pd.read_csv(f"../datasets/monks/{str(filename)}.train", sep=' ', names=col_names)
    monk_train.set_index('Id', inplace=True)
    labels = monk_train.pop('class')

    # 1-hot encoding (and transform dataframe to numpy array)
    monk_train = OneHotEncoder().fit_transform(monk_train).toarray()

    # transform labels from pandas dataframe to numpy ndarray
    labels = labels.to_numpy()[:, pd.np.newaxis]
    labels[labels == 0] = -1

    # shuffle the whole dataset once
    indexes = list(range(len(monk_train)))
    np.random.shuffle(indexes)
    monk_train = monk_train[indexes]
    labels = labels[indexes]

    return monk_train, labels


def plot_curves(tr_loss, val_loss, tr_acc, val_acc, lr, momentum, lambd, **kwargs):
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

