""" In this scripts are defined the optimizers used in the project """

from abc import ABC, abstractmethod
from losses import losses
from network import *
import numpy as np

from network import Network


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
        error = self.loss.func(predicted=net_outputs, target=target)
        deriv = self.loss.deriv(predicted=net_outputs, target=target)
        print(f"Gradient of loss function: {deriv}")

        # Chain rule
        # dE/dw = dE/d_out * d_out/d_net * d_net/d_w
        #
        # delta = d_out/d_net * dE/d_out
        # dE/dw = delta * d_net/d_w


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
    opt = optimizers['sgd'](Network(input_dim=3, units_per_layer=[1], acts=['relu']), 'mse')
    opt.optimize(inp=[0, 0, 0], target=[1])
