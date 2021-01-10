from utility import read_cup
from network.network import Network

if __name__ == '__main__':
    cup_tr_data, cup_tr_targets, cup_ts_data = read_cup()

    model = Network(
        input_dim=len(cup_tr_data[0]),
        units_per_layer=(30, 30, 2),
        acts=('identity', 'identity', 'identity'),
        init_type='random',
        lower_lim=-2.0,
        upper_lim=2.0,
    )
