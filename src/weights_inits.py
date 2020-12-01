""" Script that define different types of weights initializations """
from abc import ABC, abstractmethod


class Initialization(ABC):
    """
    This method (constructor) is unnecessary.
    Attributes are initialized in the concrete subclasses, so other methods work anyway,
    still it may be clearer to see in this form
    """
    @abstractmethod
    def __init__(self):
        self.w = 0.
        self.b = 0.
        self.type = ""

    def get_w(self):
        return self.w

    def get_b(self):
        return self.b

    def get_type(self):
        return self.type


class UniformInit(Initialization):
    def __init__(self, value=0.1, n_weights=1):
        self.w = [value] * n_weights
        self.b = value
        self.type = 'uniform'

    def set_w(self, n_weights=1):
        # self.b used as 'value' in the constructor method, they're equal
        self.w = [self.b] * n_weights


# TODO: create random one (use code commented in network)


inits = {
    'uniform': UniformInit
}

if __name__ == '__main__':
    init = inits['uniform'](n_weights=3)
    print(f"Initialization: {init.get_type()}")
    print(f"Weights: {init.get_w()}")
    print(f"Bias: {init.get_b()}")
