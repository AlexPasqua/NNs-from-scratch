import numpy as np
import argparse
from act_funcs import act_funcs
from optimizers import *
from losses import *


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
        return np.dot(inp, self.w) + self.b

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
        self.layers = []
        self.opt = None

        units = []
        # for each layer...
        for i in range(len(units_per_layer)):
            # number of weights of the units in a certain layer
            n_weights = input_dim if i == 0 else len(self.layers[i - 1].units)

            # for every unit in the current layer...
            for j in range(units_per_layer[i]):
                units.append(
                    Unit(
                        w=np.random.uniform(0., 1., n_weights),
                        b=np.random.randn() % 1.,
                        act=act_funcs[acts[i]]
                    )
                )

            self.layers.append(Layer(units=units))
            units = []

    def forward(self, inp=(2, 2, 2), verbose=False):
        """
        Performs a complete forward pass on the whole NN
        :param verbose:
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

    def compile(self, opt='testopt', loss='mse'):
        self.opt = optimizers[opt](self, loss)

    def fit(self, inp, target):
        self.opt.optimize(inp, target)

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
        type=int,
        help='The input dimension as number, e.g. 3 means input is an array of 3 elements'
    )
    parser.add_argument(
        '--inputs',
        action='store',
        nargs='+',
        type=float,
        help='Array of inputs. Length must be consistent with input dim. e.g. --inputs 2 1.5 1.1'
    )
    parser.add_argument(
        '--units_per_layer',
        action='store',
        nargs='+',
        type=int,
        help='Array as long as the number of layers. Each item is the number of units for the i-th layer e.g. 3 3 2'
    )
    parser.add_argument(
        '--act_funcs',
        action='store',
        nargs='+',
        type=str,
        help=f"List of activation function names, one for each layer. Names to be chosen among {list(act_funcs.keys())}"
    )
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    # All arguments are optional, but either they're all present or they're all None
    if (
            not all(vars(args)[arg] is None for arg in vars(args) if arg != 'verbose') and
            not all(vars(args)[arg] is not None for arg in vars(args) if arg != 'verbose')
    ):
        parser.error("All arguments are optional, but either they're all present or they're all None")

    # At this point if an arg is not None, all are not None. So we check only one argument
    if args.inputs is not None:
        # Check that lengths of arguments lists are consistent
        if args.input_dim != len(args.inputs):
            parser.error("'inputs' vector must have a length equal to 'input_dim'")
        if len(args.units_per_layer) != len(args.act_funcs):
            parser.error("'units_per_layer' vector and 'act_funcs' must have the same length")

    # Create the net object
    if args.inputs is None:  # one check is enough because either all args are None or they're all not None
        net = Network()
        net.forward(verbose=args.verbose)
    else:
        net = Network(
            input_dim=args.input_dim,
            units_per_layer=args.units_per_layer,
            acts=args.act_funcs
        )
        # net.forward(args.inputs, verbose=args.verbose)
        net.compile()
        net.fit(inp=np.array(args.inputs), target=np.array([0.25]))

    if args.verbose:
        net.print_net()

