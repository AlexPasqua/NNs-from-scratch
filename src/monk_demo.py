from network.network import Network
from model_selection import cross_valid
from utility import read_monk, plot_curves

if __name__ == '__main__':
    # read the dataset
    monk_train, labels = read_monk(name='monks-2', rescale=True)

    model_params = {
        'input_dim': 17,
        'units_per_layer': (4, 1),
        'acts': ('leaky_relu', 'tanh'),
        'init_type': 'uniform',
        'init_value': 0.2,
        'lower_lim': -0.1,
        'upper_lim': 0.1
    }
    model = Network(**model_params)

    training_params = {
        'lr': 0.4,
        'momentum': 0.8,
        #'lambd': 0.01,
        #'reg_type': 'l2',
        'lr_decay': 'exponential',
        'decay_rate': 0.95,
        'decay_steps': 500,
        'staircase': False,
        #'limit_step':200,
        'loss': 'squared',
        'opt': 'sgd',
        'epochs': 200,
        'batch_size': 'full',
        'metr': 'bin_class_acc'
    }

    # cross validation
    tr_error_values, tr_metric_values, val_error_values, val_metric_values = cross_valid(
        net=model,
        tr_val_x=monk_train,
        tr_val_y=labels,
        k_folds=5,
        **training_params
    )

    # hold-out validation
    #model.compile(opt='gd', loss='squared', metr='bin_class_acc', lr=0.2, momentum=0.6)
    #tr_error_values, tr_metric_values, val_error_values, val_metric_values = model.fit(
   #     tr_x=monk_train,
    #    tr_y=labels,
  #      epochs=600,
   #     val_split=0.,
  #      batch_size='full',
  #  )

    # plot graph
    plot_curves(
        tr_loss=tr_error_values,
        val_loss=val_error_values,
        tr_acc=tr_metric_values,
        val_acc=val_metric_values,
        **training_params
    )
