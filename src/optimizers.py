""" In this scripts are defined the optimizers used in the project """

from abc import ABC, abstractmethod
from network import Network
from act_funcs import act_funcs


class Optimizer(ABC):
    """
    Abstract class representing a generic optimizer
    (check 'ABC' documentation for more info about abstract classes in Python)

    Attributes:
        nn: Network object
    """

    def __init__(self, nn):
        self.nn = nn

    @abstractmethod
    def optimize(self):
        """
        1) Calc the error through forward pass
        2) Calc gradient of error --> partial derivs of error
        3) Chain rule for every parameter
        """



# TODO: delete this class, it's just for testing. Once there are concrete optimizers it will be pointless
class TestingConcreteClass(Optimizer, ABC):
    def __init__(self, nn):
        super(TestingConcreteClass, self).__init__(nn)
        self.nn.print_net()

    def optimize(self):
        print('optimize method')


if __name__ == '__main__':
    opt = TestingConcreteClass(
        Network(
            input_dim=2,
            units_per_layer=[1],
            acts=['relu']
        )
    )
