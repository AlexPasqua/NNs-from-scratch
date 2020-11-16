import numpy as np
from src.functions import *


class Unit:
    """
    Unit

    This class represent a unit/neuron of a NN

    Attributes:
        w: weights vector
        b: bias
        act_func: activation function
        net: weighted sum of the unit's inputs
    """

    def __init__(self, w, b, act):
        """
        Constructor
        :param w: vector of weights
        :param b: bias (number)
        """
        self.w = w
        self.b = b
        self.act = act

    def net(self, inp):
        """
        Performs weighted sum
        :param inp: unit's input vector
        :return: weighted sum of the input + bias --> dot_product(weights * input) + bias
        """
        # weighted sum + bias
        return np.dot(inp, self.w) + self.b

    def output(self, inp):
        """
        Computes activation function on the weighted sum of the input
        :return: unit's output
        """
        # compute activation function on weighted sum
        return self.act.func(self.net(inp))


class Layer:
    """
    Layer

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
    Network

    This class creates NNs

    Attributes:
        layers: list of net's layers ('Layer' objects)
    """

    def __init__(self, input_dim, units_per_layer, acts):
        """
        Constructor
        :param input_dim: the input dimension
        :param units_per_layer: list of layers' sizes as number on units
        """
        units = []
        self.layers = []

        # for each layer...
        for i in range(len(units_per_layer)):
            if i == 0:
                units_weights_length = input_dim
            else:
                units_weights_length = len(self.layers[i - 1].units)

            for j in range(units_per_layer[i]):
                units.append(
                    Unit(
                        w=np.random.uniform(0., 1., units_weights_length),
                        b=np.random.uniform(0., 1., 1),
                        act=functions[acts]
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
        x = inp   # x represents the data through the network (output of a layer, input of the next layer)
        for layer in self.layers:
            x = layer.forward_pass(x)
        return x

    def print_net(self):
        print('Neural Network:')
        for layer in self.layers:
            print('  Layer:')
            for unit in layer.units:
                print(f"\tUnit:\n\t  weights: {unit.w}\n\t  bias: {unit.b}\n\t  activation function: {unit.act.name}")
            print()


if __name__ == '__main__':
    input_dim = 3
    net = Network(input_dim=input_dim, units_per_layer=[1], acts='relu')
    inpt = [2,2,2]
    print(f"Input: {inpt}")
    net.print_net()
    print(f"Net's output: {net.forward(inpt)}")
