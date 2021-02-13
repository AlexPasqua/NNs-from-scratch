from network import Network
from model_selection import cross_valid
from utility import read_monk, plot_curves


if __name__ == '__main__':
    # MONKS DEMO

    # Create a neural network
    # input_dim must stay 17 for monks datasets
    # units_per_layer: tuple containing the number of units for each layer (except the input one)
    # init_type: 'uniform' (for random uniform) or 'fixed' to have one value for each weight
    #   limits: in init_type is 'uniform', they are the range of the weights' values
    #   if init_type is 'fixed', substitute 'limits' with 'init_value' and specify a value
    model = Network(input_dim=17, units_per_layer=(4, 1), acts=('tanh', 'tanh'), init_type='uniform', limits=(-0.2, 0.2))

    # read the dataset. Change the name in the following lines to use monks-2 or monks-3
    tr_ds_name = "monks-1.train"
    rescale = True if model.params['acts'][-1] == 'tanh' else False     # rescale the labels in [-1, +1] if needed
    monk_train, labels_tr = read_monk(name=tr_ds_name, rescale=rescale)
    monk_test, labels_ts = read_monk(name="monks-1.test", rescale=rescale)

    # Validation alternatives:
    # NOTE: do not consider the following hyperparameters as hints, they were put down very quickly.
    # NOTE: keep not commented only one alternative

    # # Alternative 1: hold-out validation
    # # compile the model (check the method definition for more info about all the accepted arguments)
    # model.compile(opt='sgd', loss='squared', metr='bin_class_acc', lr=0.3, momentum=0.8)
    # # training (check the method definition for more info about all the possible parameters)
    # tr_err, tr_metr, val_err, val_metr = model.fit(tr_x=monk_train, tr_y=labels_tr, val_split=0.15, batch_size='full',
    #                                                epochs=500, disable_tqdm=False)
    # # plot the learning curves
    # plot_curves(tr_err, val_err, tr_metr, val_metr, lbltr='Training', lblval='Validation')

    # Alternative 2: cross validation (check function's definition for info on all the accepted arguments)
    cross_valid(net=model, dataset=tr_ds_name, loss='squared', metr='bin_class_acc', lr=0.76, opt='sgd', momentum=0.83,
                epochs=500, batch_size='full', k_folds=5, disable_tqdms=(True, False), plot=True, verbose=True)


