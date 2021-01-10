from utility import read_cup, plot_curves
from network.network import Network

if __name__ == '__main__':
    # read dataset
    cup_tr_data, cup_tr_targets, cup_ts_data = read_cup()

    # create model
    model = Network(
        input_dim=len(cup_tr_data[0]),
        units_per_layer=(10, 10, 2),
        acts=('leaky_relu', 'leaky_relu', 'identity'),
        init_type='random',
        lower_lim=-0.5,
        upper_lim=0.5,
    )

    # hold-out validation
    model.compile(opt='gd', loss='squared', metr='euclidean', lr=0.3, momentum=0.6)
    tr_error_values, tr_metric_values, val_error_values, val_metric_values = model.fit(
        tr_x=cup_tr_data,
        tr_y=cup_tr_targets,
        epochs=150,
        val_split=0.2,
        batch_size='full',
    )

    # plot graph
    plot_curves(
        tr_loss=tr_error_values,
        val_loss=val_error_values,
        tr_acc=tr_metric_values,
        val_acc=val_metric_values
    )
