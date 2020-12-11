""" Script that define different types of weights initializations """
from abc import ABC, abstractmethod
import numpy as np


class Initialization(ABC):
    """
    Abstract class representing a weights initialization

    Attributes:
        w (list): weights values
        b (number): bias value
        __type: string --> type of initialization (uniform, random, etc)
    """
    @abstractmethod
    def __init__(self, w_vals, b_val, type_of_reg):
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
    """ Uniform initialization, all weights (and bias) with the same value """
    def __init__(self, val=0.1, n_weights=1, **kwargs):
        self.__check_attributes(n_weights)
        values = [val] * n_weights
        super().__init__(w_vals=values, b_val=val, type_of_reg='uniform')

    @staticmethod
    def __check_attributes(n_weights):
        if not isinstance(n_weights, int):
            raise AttributeError(f"Attribute n_weights must be a number, got {type(n_weights)}")
        if n_weights < 0:
            raise ValueError(f"Value of 'n_weights' must be >= 0. Received {n_weights}")

    # NOTE: in case it's necessary to have a setter in the subclass that calls the superclass' one:
    # @w.setter
    # def w(self, value):
    #     # the following is equivalent to: Initialization.w.fset(self, value=value)
    #     # PS: in this case the '__class__' after 'self' is necessary
    #     super(UniformInit, self.__class__).w.fset(self, value)


class RandomInit(Initialization):
    """ Random initialization, weights and bias get a random value bounded by params lower_lim and upper_lim """
    def __init__(self, n_weights=1, lower_lim=0., upper_lim=1.):
        self.__check_attributes(n_weights, lower_lim, upper_lim)
        super().__init__(w_vals=np.random.uniform(lower_lim, upper_lim, n_weights),
                         b_val=np.random.randn() % upper_lim,
                         type_of_reg='random')

    @staticmethod
    def __check_attributes(n_weights, lower_lim, upper_lim):
        if not isinstance(n_weights, int):
            raise AttributeError(f"Attribute n_weights must be a number, got {type(n_weights)}")
        if n_weights < 0:
            raise ValueError(f"Value of 'n_weights' must be >= 0. Received {n_weights}")
        if lower_lim > upper_lim:
            raise ValueError(f"'min' must be <= 'max'")


weights_inits = {
    'uniform': UniformInit,
    'random': RandomInit
}

if __name__ == '__main__':
    unif = weights_inits['uniform'](n_weights=3, val=.5)
    print(f"Initialization: {unif.type}")
    print(f"Weights: {unif.w}")
    print(f"Bias: {unif.b}\n")

    rand = weights_inits['random'](n_weights=3)
    print(f"Initialization: {rand.type}")
    print(f"Weights: {rand.w}")
    print(f"Bias: {rand.b}\n")

