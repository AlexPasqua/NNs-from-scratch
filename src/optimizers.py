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
    def __init__(self, nn, loss):
        self.nn = nn
        self.loss = losses[loss]

    @abstractmethod
    def optimize(self, inp, target):
        """
        1) Calc the error through forward pass
        2) Calc gradient of error --> partial derivs of error
        3) Chain rule for every parameter
        """
        net_outputs = self.nn.forward(inp=inp)
        err = self.loss.func(predicted=net_outputs, target=target)
        d_err = self.loss.deriv(predicted=net_outputs, target=target)

        # Scanning the layers bottom-up
        for i in range(len(self.nn.layers) - 1, -1, -1):
            curr_layer = self.nn.layers[i]
            curr_act = curr_layer.get_activation()
            if i > 0:  # if there exist a previous layer
                prev_layer = self.nn.layers[i-1]
                curr_inputs = [unit.get_out() for unit in prev_layer.get_units()]
                d_net = curr_inputs
            else:
                d_net = inp
            d_out = [curr_act.deriv(curr_unit.out) for curr_unit in curr_layer.get_units()]

            local_grads = [d_out[j] * [d_net_i for d_net_i in d_net] for j in range(len(d_out))]
            # equivalent to:
            # local_grads = []
            # local_grads.append([d_out[j] * d_net_i for j in range(len(d_out)) for d_net_i in d_net])

            # TODO: compute upstream gradient


class SGD(Optimizer, ABC):
    """
    Stochastic Gradient Descent
    """

    def __init__(self, nn, loss):
        super(SGD, self).__init__(nn, loss)

    def optimize(self, inp, target):
        Optimizer.optimize(self, inp=inp, target=target)


optimizers = {
    'sgd': SGD
}

if __name__ == '__main__':
    opt = optimizers['sgd'](Network(input_dim=3, units_per_layer=[3, 2], acts=['relu', 'relu']), 'squared')
    opt.optimize(inp=[0, 0, 0], target=[1, 1])
