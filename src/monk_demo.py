import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from network.network import Network
from model_selection import cross_valid

if __name__ == '__main__':
    parameters = {
        'input_dim': 17,
        'units_per_layer': (4, 1),
        'acts': ('leaky_relu', 'tanh'),
        'init_type': 'random',
        'weights_value': 0.2,
        'lower_lim': -0.1,
        'upper_lim': 0.1
    }
    model = Network(**parameters)

    # read the dataset
    col_names = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
    monk1_train = pd.read_csv("../datasets/monks/monks-2.train", sep=' ', names=col_names)
    monk1_train.set_index('Id', inplace=True)
    labels = monk1_train.pop('class')

    # 1-hot encoding (and transform dataframe to numpy array)
    monk1_train = OneHotEncoder().fit_transform(monk1_train).toarray()

    # transform labels from pandas dataframe to numpy ndarray
    labels = labels.to_numpy()[:, np.newaxis]
    labels[labels == 0] = -1

    # shuffle the whole dataset once
    indexes = list(range(len(monk1_train)))
    np.random.shuffle(indexes)
    monk1_train = monk1_train[indexes]
    labels = labels[indexes]

    # cross validation
    tr_error_values, tr_metric_values, val_error_values, val_metric_values = cross_valid(
        net=model,
        tr_val_x=monk1_train,
        tr_val_y=labels,
        loss='squared',
        metr='bin_class_acc',
        lr=0.3,
        lr_decay=None,
        limit_step=None,
        opt='gd',
        momentum=0.75,
        epochs=500,
        batch_size='full',
        k_folds=8
    )

    # # hold-out validation
    # model.compile(opt='gd', loss='squared', metr='bin_class_acc', lr=0.2, momentum=0.5)
    # tr_error_values, tr_metric_values, val_error_values, val_metric_values = model.fit(
    #     tr_x=monk1_train,
    #     tr_y=labels,
    #     epochs=1000,
    #     val_split=0.17,
    #     batch_size='full',
    # )

    # plot learning curve
    figure, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(range(len(tr_error_values)), tr_error_values, val_error_values)
    ax[0].set_xlabel('Epochs', fontweight='bold')
    ax[0].set_ylabel('Loss', fontweight='bold')
    ax[0].grid()
    ax[1].plot(range(len(tr_metric_values)), tr_metric_values, val_metric_values)
    ax[1].set_xlabel('Epochs', fontweight='bold')
    ax[1].set_ylabel('Accuracy', fontweight='bold')
    ax[1].grid()
    plt.show()
