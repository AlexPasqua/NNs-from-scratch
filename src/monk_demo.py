import pandas as pd
import numpy as np
from network.network import Network


if __name__ == '__main__':
    # read the dataset
    col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    monk1_train = pd.read_csv("../datasets/monks/monks-1.train", sep=' ', names=col_names)
    monk1_train.set_index('Id', inplace=True)
    labels = monk1_train.pop('class')

    # transform the dataset from pandas dataframe to numpy ndarray
    monk1_train = monk1_train.to_numpy()
    labels = labels.to_numpy()[:, np.newaxis]

    parameters = {
        'input_dim': 6,
        'units_per_layer': (3, 1),
        'acts': ('sigmoid', 'sigmoid'),
        'init_type': 'random',
        'weights_value': 0.2,
        'lower_lim': 0.0001,
        'upper_lim': 1.
    }
    model = Network(**parameters)
    model.compile(opt='gd', loss='squared', metr='bin_class_acc', lrn_rate=0.5)
    model.fit(inputs=monk1_train, targets=labels, epochs=1, batch_size=len(monk1_train))
