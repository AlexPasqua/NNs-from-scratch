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

    def optimize(self, net_inp, target):
        # ONLINE VERSION
        net_outputs = self.__nn.forward(inp=net_inp)
        err = self.loss.func(predicted=net_outputs, target=target)

        output_layer = self.__nn.layers[-1]
        output_act = output_layer.act

        # dErr_dOut: gradient of the error wrt the net's outputs
        # d_out: gradient of the net's outputs wrt the output units' weighted sums
        # dNet_dOut: gradient of the output units' weighted sum wrt the prev layer's outputs
        dErr_dOut = self.loss.deriv(predicted=net_outputs, target=target)
        d_out = [output_act.deriv(u.net) for u in output_layer.units]
        dNet_dOut = [u.w[j] for u in output_layer.units for j in range(len(u.w))]

        # delta_next is the delta of the output layer
        # 'next' is because it will be continuously overwritten with the delta of the next layer
        # it's normal mult because the vecs have the same dimension and they are 'numpy ndarray'
        delta_next = dErr_dOut * d_out

        # scan all layers from the penultimate to the first
        for i in reversed(range(len(self.__nn.layers) - 1)):
            curr_layer = self.__nn.layers[i]
            next_layer = self.__nn.layers[i + 1]

            # gradient of the error wrt the outputs of the CURRENT layer
            dErr_dOut_new = np.zeros([len(curr_layer.units)])
            for j in range(len(curr_layer.units)):
                dErr_dOut_new[j] = 0.  # it's already 0, but let's be sure
                offset = len(curr_layer.units)  # offset to select the right unit from the next layer
                for l in range(len(next_layer.units)):
                    # dNet_dOut[offset * l + j]: weight (deriv of net wrt out) on the connection j --> l
                    # equivalent to: dErr_dOut_new[j] += dErr_dOut[l] * d_out[l] * dNet_dOut[offset * l + j]
                    dErr_dOut_new[j] += dErr_dOut[l] * d_out[l] * next_layer.units[l].w[j]
            dErr_dOut = dErr_dOut_new

            # take new d_out and d_net wrt the current layer (no more the next)
            curr_act = curr_layer.act
            d_out = [curr_act.deriv(u.net) for u in curr_layer.units]
            if i > 0:
                prev_layer = self.__nn.layers[i - 1]
                d_net = prev_layer.outputs
            else:
                d_net = net_inp

            # computer delta for the current layer
            delta = [np.dot(delta_next, [u.w[j] for u in next_layer.units]) for j in range(len(curr_layer.units))]
            delta = np.multiply(delta, d_out, dtype=np.float_)
            # equivalent to:
            # delta = np.zeros([len(curr_layer.units)])
            # for j in range(len(curr_layer.units)):
            #     for l in range(len(next_layer.units)):
            #         delta[j] += next_layer.units[l].w[j] * delta_next[l]
            #     delta[j] *= d_out[j]

            # compute gradient of the error wrt this layer's weights
            if i > 0:
                prev_layer = self.__nn.layers[i - 1]
                dErr_dw = np.multiply(delta, prev_layer.outputs, dtype=np.float_)
            else:
                dErr_dw = np.multiply(delta, net_inp, dtype=np.float_)

            # weights update
            delta_w = -dErr_dw
            print(delta_w)
            print(delta)

            ########################################
            # TODO: riprendi da qui
            ########################################
            break


optimizers = {
    'sgd': SGD
}

if __name__ == '__main__':
    opt = optimizers['sgd'](Network(input_dim=3, units_per_layer=[2, 3, 2], acts=['relu', 'relu', 'relu']), 'squared')
    opt.optimize(net_inp=[0.1, 0.1, 0.1], target=[1, 1])
