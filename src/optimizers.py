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

        # 1) per step
        d_err = self.loss.deriv(predicted=net_outputs, target=target)
        for i in reversed(range(len(self.__nn.layers))):
            curr_layer = self.__nn.layers[i]
            curr_act = curr_layer.act
            d_out = [curr_act.deriv(u.net) for u in curr_layer.units]

            if i > 0:
                prev_layer = self.__nn.layers[i - 1]
                curr_inp = prev_layer.outputs
            else:
                curr_inp = net_inp

            # short version of: d_net = [curr_inp] * len(curr_layer.units)
            # because the same inputs are sent to every unit in the current layer
            # e.g. instead of storing [1,2,3, 1,2,3, 1,2,3] we store [1, 2, 3] and we know it's "repeated"
            d_net = curr_inp

            local_grads = [d_out_j * [d_net_i for d_net_i in d_net] for d_out_j in d_out]
            print(np.shape(local_grads))
            # equivalent to:
            #     # local_grads = []
            #     # local_grads.append([d_out[j] * d_net_i for j in range(len(d_out)) for d_net_i in d_net])

            # if we're on the output layer
            if i == len(self.__nn.layers) - 1:
                upstream_grad = d_err

            # TODO: finish
            else:
                pass

        # 2) formula finale diretta

        # net_outputs = self.__nn.forward(inp=net_inp)
        # err = self.loss.func(predicted=net_outputs, target=target)
        # d_err = self.loss.deriv(predicted=net_outputs, target=target)
        #
        # # Scanning the layers in a bottom-up fashion
        # for i in range(len(self.__nn.layers) - 1, -1, -1):
        #     curr_layer = self.__nn.layers[i]
        #     curr_act = curr_layer.act
        #     if i > 0:  # if there exist a previous layer
        #         prev_layer = self.__nn.layers[i - 1]
        #         # curr_inputs: inputs of the current layer's units (same for every unit in the current layer)
        #         curr_inputs = [unit.out for unit in prev_layer.units]
        #         # d_net: derivs of the weighted sum wrt the weights
        #         d_net = curr_inputs
        #     else:
        #         d_net = net_inp
        #
        #     # derivs of units' output wrt units' weighted sum
        #     d_out = [curr_act.deriv(curr_unit.out) for curr_unit in curr_layer.units]
        #
        #     # local gradients of each unit of the current layer
        #     local_grad = [d_out[j] * [d_net_i for d_net_i in d_net] for j in range(len(d_out))]
        #     # equivalent to:
        #     # local_grad = []
        #     # local_grad.append([d_out[j] * d_net_i for j in range(len(d_out)) for d_net_i in d_net])
        #
        #     # TODO: complete this for more layers
        #     # upstream gradient on units' output
        #     if i == len(self.__nn.layers) - 1:
        #         upstream_grad = d_err
        #
        #     # recompute upstream gradient wrt units' weights
        #     upstream_grad = upstream_grad * [lg for lg in local_grad]
        #
        #     # update weights
        #     curr_weights = [u.w for u in curr_layer.units]
        #     """
        #     new_weights = current_weights - learning_rate * delta_w
        #     delta_w may be:
        #         - upstream_gradient_on_the_weights
        #     """
        #     # TODO: finish


optimizers = {
    'sgd': SGD
}

if __name__ == '__main__':
    opt = optimizers['sgd'](Network(input_dim=3, units_per_layer=[3, 2], acts=['relu', 'relu']), 'squared')
    opt.optimize(net_inp=[0.1, 0.1, 0.1], target=[1, 1])
