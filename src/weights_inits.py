""" Script that define different types of weights initializations """
from abc import ABC, abstractmethod


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
        self.type = type_of_reg

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
        self.__b = value

    @type.setter
    def type(self, value):
        self.__type = value


class UniformInit(Initialization):
    def __init__(self, val=0.1, n_weights=1):
        if n_weights < 0:
            raise ValueError(f"Value of 'n_weights' must be >= 0. Received {n_weights}")
        values = [val] * n_weights
        super().__init__(w_vals=values, b_val=val, type_of_reg='uniform')

    @property
    def w(self):
        return super().w

    @property
    def b(self):
        return super().b

    @w.setter
    def w(self, value):
        # the following is equivalent to: Initialization.w.fset(self, value=value)
        # PS: in this case the '__class__' after 'self' is necessary
        super(UniformInit, self.__class__).w.fset(self, value)

    @b.setter
    def b(self, value):
        super(UniformInit, self.__class__).b.fset(self, value)


# TODO: create random one (use code commented in network)


inits = {
    'uniform': UniformInit
}

if __name__ == '__main__':
    init = inits['uniform'](n_weights=3, val=.5)
    print(f"Initialization: {init.type}")
    print(f"Weights: {init.w}")
    print(f"Bias: {init.b}\n")

    init.w = 5
    print("Set weights = 5")
    print(f"Weights: {init.w}")
    print(f"Bias: {init.b}\n")
