""" In this scripts are defined the optimizers used in the project """

from abc import ABC, abstractmethod
from losses import losses
from network import *
import numpy as np


class Optimizer(ABC):
    """
    Abstract class representing a generic optimizer
    (check 'ABC' documentation for more info about abstract classes in Python)
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
        # TODO: finish


# TODO: delete this class, it's just for testing. Once there are concrete optimizers it will be pointless
class TestingConcreteClass(Optimizer, ABC):
    def __init__(self, nn, loss):
        super(TestingConcreteClass, self).__init__(nn, loss)

    def optimize(self, inp, target):
        Optimizer.optimize(self, inp=inp, target=target)


optimizers = {
    'testopt': TestingConcreteClass
}

if __name__ == '__main__':
    opt = optimizers['testopt'](Network(), losses['mse'])
