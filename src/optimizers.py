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

        # for z in range(5):
        #     for pattern, target in zip(net_inp, targets):
        #         net_outputs = self.__nn.forward(inp=pattern)
        #         output_layer = self.__nn.layers[-1]
        #         output_act = output_layer.act
        #         n_out_units = len(output_layer.units)
        #
        #         # dErr_dOut: gradient of the error wrt the net's outputs
        #         # dOut_dNet: gradient of the net's outputs wrt the output units' weighted sums
        #         dErr_dOut = self.loss.deriv(predicted=net_outputs, target=target)
        #         dOut_dNet = [output_act.deriv(u.net) for u in output_layer.units]
        #
        #         # it's normal mult because the vecs have the same dimension and they are 'numpy ndarray'
        #         delta = -dErr_dOut * dOut_dNet
        #
        #         # will contain all the delta_weights to update the weights
        #         delta_weights = [None] * len(self.__nn.layers)
        #
        #         # retrieve output of the penultimate layer to compute the weights update of the last layer
        #         if len(self.__nn.layers) > 1:
        #             penult_layer = self.__nn.layers[-2]
        #             offset = len(penult_layer.units) + 1    # + 1 for bias
        #             dErr_dw = np.zeros([n_out_units * offset])
        #             for j in range(n_out_units):
        #                 # set dErr_dw wrt biases
        #                 dErr_dw[offset * (j+1) - 1] = 1.
        #                 # set dErr_dw wrt weights
        #                 for k in range(offset - 1):
        #                     dErr_dw[k + j * offset] = -delta[j] * penult_layer.outputs[k]
        #         else:
        #             delta_weights[-1] = [delta_j * pattern_k for delta_j in delta for pattern_k in pattern]
        #             offset = len(pattern) + 1
        #             dErr_dw = np.zeros([n_out_units * offset])
        #             for j in range(n_out_units):
        #                 dErr_dw[offset * (j+1) - 1] = 1.
        #                 for k in range(offset - 1):
        #                     dErr_dw[k + j * offset] = -delta[j] * pattern[k]
        #         delta_weights[-1] = -dErr_dw
        #         delta_next = delta
        #
        #         # scan all layers from the penultimate to the first
        #         for i in reversed(range(len(self.__nn.layers) - 1)):
        #             curr_layer = self.__nn.layers[i]
        #             next_layer = self.__nn.layers[i + 1]
        #
        #             # gradient of the error wrt the outputs of the CURRENT layer
        #             dErr_dOut_new = np.zeros([len(curr_layer.units)])
        #             for j in range(len(curr_layer.units)):
        #                 for l in range(len(next_layer.units)):
        #                     # dNet_dOut[offset * l + j]: weight (deriv of net wrt out) on the connection j --> l
        #                     dErr_dOut_new[j] += dErr_dOut[l] * dOut_dNet[l] * next_layer.units[l].w[j]
        #             dErr_dOut = dErr_dOut_new
        #
        #             # take new dOut_dNet and d_net wrt the current layer (no more the next)
        #             curr_act = curr_layer.act
        #             dOut_dNet = [curr_act.deriv(u.net) for u in curr_layer.units]
        #             if i > 0:
        #                 prev_layer = self.__nn.layers[i - 1]
        #                 curr_layer_inputs = prev_layer.outputs
        #                 d_net = prev_layer.outputs
        #             else:
        #                 d_net = pattern
        #                 curr_layer_inputs = pattern
        #
        #             # computer delta for the current layer
        #             delta = [np.dot(delta_next, [u.w[j] for u in next_layer.units]) for j in range(len(curr_layer.units))]
        #             delta = np.multiply(delta, dOut_dNet, dtype=np.float_)
        #             delta_next = delta
        #             # equivalent to:
        #             # delta = np.zeros([len(curr_layer.units)])
        #             # for j in range(len(curr_layer.units)):
        #             #     for l in range(len(next_layer.units)):
        #             #         delta[j] += next_layer.units[l].w[j] * delta_next[l]
        #             #     delta[j] *= dOut_dNet[j]
        #
        #             # compute gradient of the error wrt this layer's weights
        #             dErr_dw = np.zeros([len(curr_layer.units) * len(curr_layer_inputs)])
        #             offset = len(curr_layer_inputs)
        #             for j in range(len(curr_layer.units)):
        #                 for k in range(len(curr_layer_inputs)):
        #                     dErr_dw[k + j * offset] = curr_layer_inputs[k] * delta[j]
        #
        #             delta_weights[i] = -dErr_dw
        #
        #         # weights update
        #         for i in range(len(self.__nn.layers)):
        #             curr_layer = self.__nn.layers[i]
        #             if i == 0:
        #                 curr_layer_inputs = pattern
        #             else:
        #                 prev_layer = self.__nn.layers[i - 1]
        #                 curr_layer_inputs = prev_layer.outputs
        #             offset = len(curr_layer_inputs)
        #             curr_layer.weights_biases += self.lrn_rate * delta_weights[i]
        #             # equivalent to:
        #             # for j in range(len(curr_layer.units)):
        #             #     for k in range(len(curr_layer.units[j].w)):
        #             #         curr_layer.units[j].w[k] += self.lrn_rate * delta_weights[i][k + j * offset]
        #
        #         # net_outputs = self.__nn.forward(inp=pattern)
        #         # err = self.loss.func(predicted=net_outputs, target=target)
        #         # print(np.sum(err) / len(err), '\n')


optimizers = {
    'sgd': SGD
}

if __name__ == '__main__':
    opt = optimizers['sgd'](
        Network(
            input_dim=3,
            units_per_layer=[2],
            acts=['sigmoid'],
            weights_init='uniform',
            weights_value=0.5
        ),
        loss='squared')
    opt.optimize(net_inp=np.array([0.1, 0.1, 0.1]),
                 targets=np.array([0.99, 0.99]))
