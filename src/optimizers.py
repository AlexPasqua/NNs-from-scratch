""" In this scripts are defined the optimizers used in the project """

from abc import ABC, abstractmethod
from losses import losses
from network import *


class Optimizer(ABC):
    """
    Abstract class representing a generic optimizer
    (check 'ABC' documentation for more info about abstract classes in Python)
    """

    def __init__(self, nn, loss):
        self.nn = nn
        self.loss = losses[loss]

    @abstractmethod
    def optimize(self):
        """
        1) Calc the error through forward pass
        2) Calc gradient of error --> partial derivs of error
        3) Chain rule for every parameter
        """


# TODO: delete this class, it's just for testing. Once there are concrete optimizers it will be pointless
class TestingConcreteClass(Optimizer, ABC):
    def __init__(self, nn, loss):
        super(TestingConcreteClass, self).__init__(nn, loss)

    def optimize(self, inp):
        print('optimize method')
        net_outputs = self.nn.forward(inp)
        print(net_outputs)


optimizers = {
    'testopt': TestingConcreteClass
}

if __name__ == '__main__':
    opt = optimizers['testopt'](Network(), losses['mse'])
