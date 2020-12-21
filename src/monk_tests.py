""" Script to do tests with the Monk dataset """
# from typing import List
# from numpy.core._multiarray_umath import ndarray
# import pandas as pd
import numpy as np

from network import Network
import csv

if __name__ == '__main__':
    # read the dataset
    # col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    # monk1_train = pd.read_csv("../datasets/monks/monks-1.train", sep=' ', names=col_names)
    # monk1_train.set_index('Id', inplace=True)
    # labels = monk1_train.pop('class')

    # transform the dataset from pandas dataframe to numpy ndarray
    # monk1_train = monk1_train.to_numpy()
    # labels = labels.to_numpy()[:, np.newaxis]

    labels = []
    monk1_train = []

    with open("../datasets/monks/monks-1.train") as infile:
        reader = csv.reader(infile, delimiter=" ")
        for row in reader:
            labels.append([int(row[1])])

            # One-Hot encoding
            data = np.zeros(17)
            data[int(row[2]) - 1] = 1
            data[int(row[3]) + 2] = 1
            data[int(row[4]) + 5] = 1
            data[int(row[5]) + 7] = 1
            data[int(row[6]) + 10] = 1
            data[int(row[7]) + 14] = 1

            monk1_train.append(data)

    parameters = {
        'input_dim': 17,
        'units_per_layer': (5, 1),
        'acts': ('relu', 'sigmoid'),
        'weights_init': 'random'

    }

    model = Network(**parameters)
    model.print_net()
    model.compile(opt='sgd', loss='squared', metr='class_acc', lrn_rate=0.8)
    model.fit(inp=monk1_train, target=labels, epochs=40, batch_size=1)
