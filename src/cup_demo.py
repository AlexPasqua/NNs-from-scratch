from utility import read_cup, plot_curves, get_best_models
from network import Network
from model_selection import grid_search, cross_valid
from multiprocessing import Process, Manager
from datetime import datetime
import numpy as np
import os


def holdout_validation(pnum, cup_tr_data, cup_tr_targets, res):
    # create model
    model = Network(
        input_dim=len(cup_tr_data[0]),
        units_per_layer=(10, 10, 2),
        acts=('leaky_relu', 'leaky_relu', 'identity'),
        init_type='uniform',
        limits=(0.0001, 0.001)
    )

    # hold-out validation
    model.compile(opt='sgd', loss='squared', metr='euclidean', lr=0.0002, momentum=0.6)
    tr_error_values, tr_metric_values, val_error_values, val_metric_values = model.fit(
        tr_x=cup_tr_data,
        tr_y=cup_tr_targets,
        epochs=80,
        val_split=0.2,
        batch_size=30,
        disable_tqdm=True
    )
    # print(f"Final values:\nTR loss: {tr_error_values[-1]}\tTR metric: {tr_metric_values[-1]}")
    # print(f"VAL loss: {val_error_values[-1]}\tVAL metric: {val_metric_values[-1]}\n")
    res[pnum] = {'tr_err': tr_metric_values[-1], 'tr_metr': tr_metric_values[-1],
                 'val_err': val_error_values[-1], 'val_metr': val_metric_values[-1]}


if __name__ == '__main__':
    # read dataset
    cup_tr_data, cup_tr_targets, cup_ts_data = read_cup()

    # create model
    model = Network(
        input_dim=len(cup_tr_data[0]),
        units_per_layer=(10, 10, 2),
        acts=('leaky_relu', 'leaky_relu', 'identity'),
        init_type='uniform',
        limits=(0.0001, 0.001)
    )

    # with Manager() as manager:
    #     res = manager.dict()
    #     processes = []
    #     kwargs = {'pnum': 0, 'cup_tr_data': cup_tr_data, 'cup_tr_targets': cup_tr_targets, 'res': res}
    #     for i in range(os.cpu_count() - 1):
    #         print('registering process %d' % i)
    #         kwargs['pnum'] = i
    #         processes.append(Process(target=holdout_validation, kwargs=kwargs))
    #
    #     for process in processes:
    #         process.start()
    #
    #     for j in range(len(processes)):
    #         processes[j].join()
    #
    #     final_res = {'avg_tr_err': np.mean([res[k]['tr_err'] for k in res.keys()]),
    #                  'avg_tr_metr': np.mean([res[k]['tr_metr'] for k in res.keys()]),
    #                  'avg_val_err': np.mean([res[k]['val_err'] for k in res.keys()]),
    #                  'avg_val_metr': np.mean([res[k]['val_metr'] for k in res.keys()])}
    #
    #     for k, v in final_res.items():
    #         print(k, ': ', v)

    # cross validation
    # cross_valid(model, "cup", 'squared', 'euclidean', lr=0.002, momentum=0.6, epochs=15,
    #             batch_size=30, k_folds=5, disable_tqdms=(True, False), verbose=True)

    # grid search
    gs_params = {'units_per_layer': ((2, 2),),
                 'acts': (('leaky_relu', 'identity'),),
                 'init_type': ('uniform',),
                 'limits': ((-0.5, 0.5), (-0.001, 0.001)),
                 'momentum': (0.0,),
                 'batch_size': ('full',),
                 'lr': (0.002, 0.5),
                 'loss': ('squared',),
                 'metr': ('euclidean',),
                 'epochs': (20,)}
    grid_search(dataset="cup", params=gs_params)
    best_model, params = get_best_models("cup", 1)
    best_model = best_model[0]
    params = params[0]
    best_model.print_topology()
    best_model.compile(opt='sgd', **params)
    tr_error_values, tr_metric_values, val_error_values, val_metric_values = best_model.fit(
        tr_x=cup_tr_data, tr_y=cup_tr_targets, disable_tqdm=False, **params)

    # plot graph
    plot_curves(
        tr_loss=tr_error_values,
        val_loss=val_error_values,
        tr_acc=tr_metric_values,
        val_acc=val_metric_values
    )
