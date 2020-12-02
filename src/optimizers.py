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
    def __init__(self, nn, loss, lrn_rate=0.01):
        self.nn = nn
        self.loss = losses[loss]
        self.lrn_rate = lrn_rate

    def optimize(self, net_inp, target):
        """
        1) Calc the error through forward pass
        2) Calc gradient of error --> partial derivs of error
        3) Chain rule for every parameter
        """
        net_outputs = self.nn.forward(inp=net_inp)
        err = self.loss.func(predicted=net_outputs, target=target)
        d_err = self.loss.deriv(predicted=net_outputs, target=target)

        # Scanning the layers in a bottom-up fashion
        for i in range(len(self.nn.layers) - 1, -1, -1):
            curr_layer = self.nn.layers[i]
            curr_act = curr_layer.get_activation()
            if i > 0:  # if there exist a previous layer
                prev_layer = self.nn.layers[i - 1]
                # curr_inputs: inputs of the current layer's units (same for every unit in the current layer)
                curr_inputs = [unit.get_out() for unit in prev_layer.get_units()]
                # d_net: derivs of the weighted sum wrt the weights
                d_net = curr_inputs
            else:
                d_net = net_inp

            # derivs of units' output wrt units' weighted sum
            d_out = [curr_act.deriv(curr_unit.out) for curr_unit in curr_layer.get_units()]

            # local gradients of each unit of the current layer
            local_grads = [d_out[j] * [d_net_i for d_net_i in d_net] for j in range(len(d_out))]
            # equivalent to:
            # local_grads = []
            # local_grads.append([d_out[j] * d_net_i for j in range(len(d_out)) for d_net_i in d_net])

            # TODO: complete this for more layers
            # upstream gradient on units' output
            if i == 0:
                upstream_grad = d_err

            # recompute upstream gradient wrt units' weights
            upstream_grad = upstream_grad * [lg for lg in local_grads]

            # update weights
            curr_weights = [u.w for u in curr_layer.get_units()]
            print(curr_weights)
            # TODO: finish


class SGD(Optimizer, ABC):
    """
    Stochastic Gradient Descent
    """

    def __init__(self, nn, loss, lrn_rate=0.01):
        super(SGD, self).__init__(nn, loss, lrn_rate)

    def optimize(self, net_inp, target):
        Optimizer.optimize(self, net_inp=net_inp, target=target)


optimizers = {
    'sgd': SGD
}

if __name__ == '__main__':
    opt = optimizers['sgd'](Network(input_dim=3, units_per_layer=[1], acts=['relu']), 'squared')
    opt.optimize(net_inp=[0.1, 0.1, 0.1], target=[1])
