""" In this scripts are defined the optimizers used in the project """

from abc import ABC, abstractmethod
from losses import losses
from network import *
import numpy as np


class Optimizer(ABC):
    """
    Abstract class representing a generic optimizer
    (check 'ABC' documentation for more info about abstract classes in Python)

    Attributes:
        nn: Neural Network --> 'Network' object
        loss: loss function --> 'Function' object
    """

    @abstractmethod
    def __init__(self, loss, lrn_rate=0.01):
        self.__loss = losses[loss]
        self.__lrn_rate = lrn_rate

    @property
    def loss(self):
        return self.__loss

    @property
    def lrn_rate(self):
        return self.__lrn_rate

    def optimize(self, net_inp, target):
        pass


class SGD(Optimizer, ABC):
    """ Stochastic Gradient Descent """

    def __init__(self, nn, loss, lrn_rate=0.01):
        self.__nn = nn
        super(SGD, self).__init__(loss, lrn_rate)
        # makes sure lrn_rate is a value between 0 and 1
        if lrn_rate <= 0 or lrn_rate > 1:
            raise ValueError('lrn_rate should be a value between 0 and 1, Got:{}'.format(lrn_rate))

    def optimize(self, net_inp, targets):
        """
        :param net_inp: (numpy ndarray) inputs
        :param targets:
        :return:
        """
        # ONLINE VERSION
        if len(net_inp.shape) < 2:
            net_inp = net_inp[np.newaxis, :]
        if len(targets.shape) < 2:
            targets = targets[np.newaxis, :]

        for pattern, target in zip(net_inp, targets):
            output_layer = self.__nn.layers[-1]
            output_act = output_layer.act
            net_outputs = self.__nn.forward(inp=pattern)

            dErr_dOut = self.loss.deriv(predicted=net_outputs, target=target)
            dOut_dNet = [output_act.deriv(u.net) for u in output_layer.units]
            delta = -dErr_dOut * dOut_dNet
            delta_next = delta

            # retrieve the inputs of the output layer to compute the weights update for the output layer
            out_layer_inputs = self.__nn.layers[-2].outputs if len(self.__nn.layers) > 1 else pattern
            dErr_dBiases = -delta
            dErr_dWeights = [
                -delta[j] * out_layer_inputs[i]
                for j in range(len(delta))
                for i in range(len(out_layer_inputs))
            ]

            # variables used for weights and biases updates
            # delta_weights: list of lists --> layers x weights_in_layer
            # delta_biases: list of lists --> layers x biases_in_layer
            # delta_weights = delta_biases = [[]] * len(self.__nn.layers)
            delta_weights = [[]] * len(self.__nn.layers)
            delta_biases = [[]] * len(self.__nn.layers)
            delta_weights[-1] = [-dErr_dWeights[i] for i in range(len(dErr_dWeights))]
            delta_biases[-1] = [-dErr_dBiases[j] for j in range(len(dErr_dBiases))]

            # scan all layers from the penultimate to the first
            for layer_index in reversed(range(len(self.__nn.layers) - 1)):
                curr_layer = self.__nn.layers[layer_index]
                next_layer = self.__nn.layers[layer_index + 1]
                n_curr_units = len(curr_layer.units)    # number of units in the current layer
                n_next_units = len(next_layer.units)    # number of units in the next layer
                curr_act = curr_layer.act

                dOut_dNet = [curr_act.deriv(u.net) for u in curr_layer.units]

                delta = [np.dot(delta_next, [u.w[j] for u in next_layer.units]) for j in range(n_curr_units)]
                delta = np.multiply(delta, dOut_dNet)
                delta_next = delta
                # equivalent to:
                # delta = np.zeros([len(curr_layer.units)])
                # for j in range(len(curr_layer.units)):
                #     for l in range(len(next_layer.units)):
                #         delta[j] += next_layer.units[l].w[j] * delta_next[l]
                #     delta[j] *= dOut_dNet[j]

                curr_layer_inputs = self.__nn.layers[layer_index - 1].outputs if layer_index > 1 else pattern
                dErr_dBiases = -delta
                dErr_dWeights = [
                    -delta[j] * curr_layer_inputs[i]
                    for j in range(len(delta))
                    for i in range(len(curr_layer_inputs))
                ]
                delta_weights[layer_index] = [-dErr_dWeights[i] for i in range(len(dErr_dWeights))]
                delta_biases[layer_index] = [-dErr_dBiases[j] for j in range(len(dErr_dBiases))]

            # update weights and biases
            for layer_index in range(len(self.__nn.layers)):
                curr_layer = self.__nn.layers[layer_index]
                curr_layer.weights = delta_weights[layer_index]
                curr_layer.biases = delta_biases[layer_index]


optimizers = {
    'sgd': SGD
}

if __name__ == '__main__':
    opt = optimizers['sgd'](
        Network(
            input_dim=3,
            units_per_layer=[3, 2, 2],
            acts=['sigmoid', 'sigmoid', 'sigmoid'],
            weights_init='uniform',
            weights_value=0.5
        ),
        loss='squared')
    opt.optimize(net_inp=np.array([0.1, 0.1, 0.1]),
                 targets=np.array([0.5, 0.5]))
