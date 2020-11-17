import numpy as np
import argparse
from functions import *


class Unit:
    """
    This class represent a unit/neuron of a NN

    Attributes:
        w: weights vector
        b: bias
        act: activation function
    """

    def __init__(self, w, b, act):
        """
        Constructor
        :param w: vector of weights
        :param b: bias (number)
        :param act: activation function --> 'Function' obj (see 'functions.py')
        """
        self.w = w
        self.b = b
        self.act = act

    def net(self, inp):
        """
        Performs weighted sum
        :param inp: unit's input vector
        :return: weighted sum of the input + bias
        """
        # return np.dot(inp, self.w) + self.b    # TODO: does not work, why?

        # weighted sum + bias
        weighted_sum = 0
        for i in range(len(inp)):
            weighted_sum += inp[i] * self.w[i]
        return weighted_sum + self.b

    def output(self, inp):
        """
        Computes activation function on the weighted sum of the input
        :param inp: unit's input vector
        :return: unit's output
        """
        # compute activation function on weighted sum
        return self.act.func(self.net(inp))


class Layer:
    """
    This class implements a layer of a NN

    Attributes:
        units: list of layer's units ('Unit' objects)
    """

    def __init__(self, units):
        """
        Constructor
        :param units: list on layer's units ('Unit' objects)
        """
        self.units = units

    def forward_pass(self, inp):
        """
        Performs the forward pass on the current layer
        :param inp: input vector
        :return: the vector of the current layer's soutputs
        """
        outputs = []
        for unit in self.units:
            outputs.append(unit.output(inp))

        return outputs


class Network:
    """
    This class creates NNs

    Attributes:
        layers: list of net's layers ('Layer' objects)
    """

    def __init__(self, input_dim=3, units_per_layer=(3, 2), acts=('relu', 'sigmoid')):
        """
        Constructor
        :param input_dim: the input dimension
        :param units_per_layer: list of layers' sizes as number on units
        :param acts: list of activation function names (one for each layer)
        """
        units = []
        self.layers = []

        # for each layer...
        for i in range(len(units_per_layer)):
            # number of weights of the units in a certain layer
            n_weights = input_dim if i == 0 else len(self.layers[i - 1].units)

            # for every unit in the current layer...
            for j in range(units_per_layer[i]):
                units.append(
                    Unit(
                        w=np.random.uniform(0., 1., n_weights),
                        b=np.random.uniform(0., 1., 1),
                        act=functions[acts[i]]
                    )
                )

            self.layers.append(Layer(units=units))
            units = []

    def forward(self, inp=(2, 2, 2), verbose=False):
        """
        Performs a complete forward pass on the whole NN
        :param inp: net's input vector
        :return: net's output
        """
        if verbose:
            print(f"Net's inputs: {inp}")
        # x represents the data through the network (output of a layer, input of the next layer)
        x = inp
        for layer in self.layers:
            x = layer.forward_pass(x)
        if verbose:
            print(f"Net's output: {x}")
        return x

    def print_net(self):
        """
        Prints the network's architecture and parameters
        """
        print('Neural Network:')
        for layer in self.layers:
            print('  Layer:')
            for unit in layer.units:
                print(f"\tUnit:\n\t  weights: {unit.w}\n\t  bias: {unit.b}\n\t  activation function: {unit.act.name}")
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates a NN with specified topology, inputs and activation functions'
    )
    parser.add_argument(
        '--input_dim',
        action='store',
        help='The input dimension as number, e.g. 3 means input is an array of 3 elements'
    )
    parser.add_argument(
        '--inputs',
        action='store',
        help='Array of inputs. Length must be consistent with input dim'
    )
    parser.add_argument(
        '--units_per_layer',
        action='store',
        help='Array as long as the number of layers. Each item is the number of units for the i-th layer'
    )
    parser.add_argument(
        '--activation_functions',
        action='store',
        help="List of activation function names, one for each layer. Names to be chosen among {'relu', 'sigmoid'}"
    )
    args = parser.parse_args()

    # All arguments are optional, but either they're all present or they're all None
    none_found = init_found = False
    for var in vars(args):
        if vars(args)[var] is None:
            none_found = True
        if vars(args)[var] is not None:
            init_found = True

        print(len(vars(args)['inputs']))

    if none_found and init_found:
        parser.error("All arguments are optional, but either they're all present or they're all None")

    # the 1st check is enough: at this point is an arg is not None, all are not None
    if args.inputs is not None and args.input_dim != len(args.inputs):
        parser.error("'inputs' vector must have a length equal to 'input_dim'")

    # Create the net object
    if args.inputs is None:  # one check is enough because either all args are None or they're all not None
        net = Network()
        net.print_net()
        net.forward(verbose=True)
    else:
        net = Network(
            input_dim=args.input_dim,
            units_per_layer=args.units_per_layer,
            acts=args.activation_functions
        )
        net.print_net()
        net.forward(args.input, verbose=True)
