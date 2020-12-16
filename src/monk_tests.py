""" Script to do tests with the Monk dataset """
from network import Network

if __name__ == '__main__':
    parameters = {
        'input_dim': 3,
        'units_per_layer': (3, 2),
        'acts': ('relu', 'sigmoid'),
        'weights_init': 'uniform',
        'weights_value': 0.1
    }
    model = Network(**parameters)
    model.print_net()