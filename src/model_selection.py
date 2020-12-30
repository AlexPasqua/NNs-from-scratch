import numpy as np
from network.network import Network


def cross_valid(net, inputs, targets, epochs=1, batch_size=1, k_folds=5):
    # split the dataset into folds
    x_folds = np.array(np.array_split(inputs, k_folds), dtype=object)
    y_folds = np.array(np.array_split(targets, k_folds), dtype=object)

    # initialize vectors for plots
    tr_error_values, tr_metric_values = np.zeros(epochs), np.zeros(epochs)
    val_error_values, val_metric_values = np.zeros(epochs), np.zeros(epochs)

    # CV cycle
    for i in range(k_folds):
        # create validation set and training set using the folds
        valid_set = x_folds[i]
        valid_targets = y_folds[i]
        train_folds = np.concatenate((x_folds[: i], x_folds[i + 1:]))
        target_folds = np.concatenate((y_folds[: i], y_folds[i + 1:]))
        train_set = train_folds[0]
        train_targets = target_folds[0]
        for j in range(1, len(train_folds)):
            train_set = np.concatenate((train_set, train_folds[j]))
            train_targets = np.concatenate((train_targets, target_folds[j]))

        # training
        # TODO: pass parameters of compile to "cross_valid" function
        net.compile(opt='gd',
                    loss='squared',
                    metr='bin_class_acc',
                    lr=0.3,
                    # lr_decay='linear',
                    # limit_step=100,
                    momentum=0.8)
        tr_err, tr_metric, val_err, val_metric = net.fit(tr_x=train_set,
                                                         tr_y=train_targets,
                                                         val_x=valid_set,
                                                         val_y=valid_targets,
                                                         epochs=epochs,
                                                         batch_size=batch_size)
        tr_error_values += tr_err
        tr_metric_values += tr_metric
        val_error_values += val_err
        val_metric_values += val_metric

        # reset net's weights and compile the "new" model
        net = Network(**net.params)

    # average the validation results of every fold
    tr_error_values /= float(k_folds)
    tr_metric_values /= float(k_folds)
    val_error_values /= float(k_folds)
    val_metric_values /= float(k_folds)
    return tr_error_values, tr_metric_values, val_error_values, val_metric_values
