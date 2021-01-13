import numpy as np
import tqdm
from multiprocessing import Process
from utility import plot_curves
from network import Network


def cross_valid(net, tr_val_x, tr_val_y, loss, metr, lr, lr_decay=None, limit_step=None, opt='gd', momentum=0.,
                epochs=1, batch_size=1, k_folds=5, reg_type='l2', lambd=0):
    # split the dataset into folds
    x_folds = np.array(np.array_split(tr_val_x, k_folds), dtype=object)
    y_folds = np.array(np.array_split(tr_val_y, k_folds), dtype=object)

    # initialize vectors for plots
    tr_error_values, tr_metric_values = np.zeros(epochs), np.zeros(epochs)
    val_error_values, val_metric_values = np.zeros(epochs), np.zeros(epochs)
    val_metr_for_fold, val_err_for_fold = [], []

    # CV cycle
    for i in tqdm.tqdm(range(k_folds), desc='Iterating over folds', disable=False):
        # create validation set and training set using the folds (for one iteration of CV)
        val_data, val_targets = x_folds[i], y_folds[i]
        tr_data_folds = np.concatenate((x_folds[: i], x_folds[i + 1:]))
        tr_targets_folds = np.concatenate((y_folds[: i], y_folds[i + 1:]))
        # here tr_data_folds & tr_targets_folds are still a "list of folds", we need a single seq as a whole
        tr_data = tr_data_folds[0]
        tr_targets = tr_targets_folds[0]
        for j in range(1, len(tr_data_folds)):
            tr_data = np.concatenate((tr_data, tr_data_folds[j]))
            tr_targets = np.concatenate((tr_targets, tr_targets_folds[j]))

        # compile and fit the model on the current training set and evaluate it on the current validation set
        net.compile(opt=opt, loss=loss, metr=metr, lr=lr, lr_decay=lr_decay, limit_step=limit_step, momentum=momentum,
                    reg_type=reg_type, lambd=lambd)
        tr_history = net.fit(tr_x=tr_data, tr_y=tr_targets, val_x=val_data, val_y=val_targets, epochs=epochs,
                             batch_size=batch_size)

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
        val_metr_for_fold.append(tr_history[3][-1])
        val_err_for_fold.append(tr_history[3][-1])

        # reset net's weights for the next iteration of CV
        net = Network(**net.params)

    # average the validation results of every fold
    tr_error_values /= k_folds
    tr_metric_values /= k_folds
    val_error_values /= k_folds
    val_metric_values /= k_folds

    # print k-fold metrics
    print("\nValidation scores per fold:")
    for i in range(k_folds):
        print(f"Fold {i + 1} - Loss: {val_err_for_fold[i]} - Accuracy: {val_metr_for_fold[i]}")
        print("--------------------------------------------------------------")
    print('\nAverage validation scores for all folds:')
    print(
        f"Loss: {np.mean(val_err_for_fold)} - std:(+/- {np.std(val_err_for_fold)})\nAccuracy: {np.mean(val_metr_for_fold)} - std:(+/- {np.std(val_metr_for_fold)})")

    plot_curves(tr_error_values, val_error_values, tr_metric_values, val_metric_values)

    return tr_error_values, tr_metric_values, val_error_values, val_metric_values


def grid_search(dev_set_x, dev_set_y):
    grid_search_params = {
        'units_per_layer': ((10, 10, 2),),
        'acts': (('leaky_relu', 'leaky_relu', 'identity'),),
        'momentum': (0.6,),
        'batch_size': (30,),
        'lr': (0.0002, 0.001),
        'init_type': ('uniform',),
        'lower_lim': (0.0001,),
        'upper_lim': (0.001,)
    }

    processes = []
    for units_per_layer in grid_search_params['units_per_layer']:
        for acts in grid_search_params['acts']:
            for momentum in grid_search_params['momentum']:
                for batch_size in grid_search_params['batch_size']:
                    for lr in grid_search_params['lr']:
                        for init_type in grid_search_params['init_type']:
                            for lower_lim in grid_search_params['lower_lim']:
                                for upper_lim in grid_search_params['upper_lim']:
                                    # print(units_per_layer, acts, momentum, batch_size, lr, init_type, lower_lim, upper_lim)
                                    model = Network(
                                        input_dim=len(dev_set_x[0]),
                                        units_per_layer=units_per_layer,
                                        acts=acts,
                                        init_type=init_type,
                                        lower_lim=lower_lim,
                                        upper_lim=upper_lim,
                                    )

                                    processes.append(Process(target=cross_valid, kwargs={
                                        'net': model,
                                        'tr_val_x': dev_set_x,
                                        'tr_val_y': dev_set_y,
                                        'loss': 'squared',
                                        'metr': 'euclidean',
                                        'lr': lr,
                                        'momentum': momentum,
                                        'epochs': 5,
                                        'batch_size': batch_size,
                                        'k_folds': 10
                                    }))

                                    # cross_valid(model, dev_set_x, dev_set_y, 'squared', 'euclidean', lr=lr,
                                    #             momentum=momentum, epochs=150, batch_size=batch_size, k_folds=10)

    for process in processes:
        process.start()

    for process in processes:
        process.join()
        # print('ended')
