import numpy as np
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

    def __init__(self, input_dim, units_per_layer, acts):
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

    def forward(self, inp):
        """
        Performs a complete forward pass on the whole NN
        :param inp: net's input vector
        :return: net's output
        """
        # x represents the data through the network (output of a layer, input of the next layer)
        x = inp
        for layer in self.layers:
            x = layer.forward_pass(x)
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
    input_dim = 3
    net = Network(input_dim=input_dim, units_per_layer=[2, 1], acts=['relu', 'sigmoid'])
    inpt = [2, 2, 2]
    print(f"Input: {inpt}")
    net.print_net()
    print(f"Net's output: {net.forward(inpt)}")
