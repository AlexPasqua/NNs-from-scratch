import numpy as np


class Unit:
    """
    Unit

    This class represent a unit/neuron of a NN

    Attributes:
        w: weights vector
        b: bias
        net: weighted sum of the unit's inputs
    """

    def __init__(self, w, b):
        """
        Constructor
        :param w: vector of weights
        :param b: bias (number)
        """
        self.w = w
        self.b = b

    def net(self, inp):
        """
        Performs weighted sum
        :param inp: unit's input vector
        :return: weighted sum of the input + bias --> dot_product(weights * input) + bias
        """
        # weighted sum + bias
        self.net = np.dot(inp, self.w) + self.b

    def output(self):
        """
        Computes activation function on the weighted sum of the input
        :return: unit's output
        """
        # compute activation function on weighted sum
        # TODO: complete
        return 0


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
            outputs.append(unit.output())

        return outputs


class Network:
    """
    Network

    This class creates NNs

    Attributes:
        layers: list of net's layers ('Layer' objects)
    """

    def __init__(self, input_dim, units_per_layer):
        """
        Constructor
        :param input_dim: the input dimension
        :param units_per_layer: list of layers' sizes as number on units
        """
        units = []
        self.layers = []
        for i in range(len(units_per_layer)):
            if i == 0:
                units_weights_length = input_dim
            else:
                units_weights_length = len(self.layers[i - 1].units)

            for j in range(units_per_layer[i]):
                units.append(
                    Unit(
                        w=np.random.uniform(0., 1., units_weights_length),
                        b=np.random.uniform(0., 1., 1)
                    )
                )

            self.layers.append(Layer(units=units))
            units = []

    def forward(self, input):
        """
        Performs a complete forward pass on the whole NN
        :param input: net's input vector
        :return: net's output
        """
        layer_input = input
        for layer in self.layers:
            layer_input = layer.forward_pass(layer_input)

        return layer_input  # it's actually the net's output


if __name__ == '__main__':
    input_dim = 9
    net = Network(input_dim=input_dim, units_per_layer=[3, 2])
    print(f"Net's output: {net.forward(np.random.randn(input_dim))}")
