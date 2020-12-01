""" Script that define different types of weights initializations """
from abc import ABC, abstractmethod
import numpy as np


class Initialization(ABC):
    """
    This method (constructor) is unnecessary.
    Attributes are initialized in the concrete subclasses, so other methods work anyway,
    still it may be clearer to see in this form
    """
    @abstractmethod
    def __init__(self, w_vals=(0.1,), b_val=(0.1,), type_of_reg=""):
        self.w = w_vals
        self.b = b_val
        self.__type = type_of_reg

    @property
    def w(self):
        return self.__w

    @property
    def b(self):
        return self.__b

    @property
    def type(self):
        return self.__type

    @w.setter
    def w(self, value):
        self.__w = value

    @b.setter
    def b(self, value):
        if isinstance(value, list):
            raise AttributeError("Parameter b (bias) must be a number, not a list")
        self.__b = value


class UniformInit(Initialization):
    def __init__(self, val=0.1, n_weights=1):
        if not isinstance(n_weights, int):
            raise AttributeError(f"Attribute n_weights must be a number, got {type(n_weights)}")
        if n_weights < 0:
            raise ValueError(f"Value of 'n_weights' must be >= 0. Received {n_weights}")
        values = [val] * n_weights
        super().__init__(w_vals=values, b_val=val, type_of_reg='uniform')

    # NOTE: in case it's necessary to have a setter in the subclass that calls the superclass' one:
    # @w.setter
    # def w(self, value):
    #     # the following is equivalent to: Initialization.w.fset(self, value=value)
    #     # PS: in this case the '__class__' after 'self' is necessary
    #     super(UniformInit, self.__class__).w.fset(self, value)


class RandomInit(Initialization):
    def __init__(self, n_weights=1, min=0., max=1.):
        if not isinstance(n_weights, int):
            raise AttributeError(f"Attribute n_weights must be a number, got {type(n_weights)}")
        if n_weights < 0:
            raise ValueError(f"Value of 'n_weights' must be >= 0. Received {n_weights}")
        if min > max:
            raise ValueError(f"'min' must be <= 'max'")
        super().__init__(w_vals=np.random.uniform(min, max, n_weights),
                         b_val=np.random.randn() % max,
                         type_of_reg='random')


inits = {
    'uniform': UniformInit,
    'random': RandomInit
}

if __name__ == '__main__':
    unif = inits['uniform'](n_weights=3, val=.5)
    print(f"Initialization: {unif.type}")
    print(f"Weights: {unif.w}")
    print(f"Bias: {unif.b}\n")

    rand = inits['random'](n_weights=3)
    print(f"Initialization: {rand.type}")
    print(f"Weights: {rand.w}")
    print(f"Bias: {rand.b}\n")

