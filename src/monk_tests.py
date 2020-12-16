""" Script to do tests with the Monk dataset """
import pandas as pd
import numpy as np
from network import Network


if __name__ == '__main__':
    # read the dataset
    col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    monk1_train = pd.read_csv("../datasets/monks/monks-1.train", sep=' ', names=col_names)
    monk1_train.set_index('Id', inplace=True)
    labels = monk1_train.pop('class')
    # print(monk1_train.describe().T)

    # transform the dataset from pandas dataframe to numpy ndarray
    monk1_train = monk1_train.to_numpy()
    labels = labels.to_numpy()

    parameters = {
        'input_dim': 3,
        'units_per_layer': (3, 2),
        'acts': ('relu', 'sigmoid'),
        'weights_init': 'uniform',
        'weights_value': 0.1
    }
    model = Network(**parameters)
    # model.print_net()
