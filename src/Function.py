""" Script containing the definition of the class representing a function"""


class Function:
    """
    Class representing a function

    Attributes:
        func (function "pointer"): Represents the function itself
        name (string): name of the function
    """
    def __init__(self, func, name):
        self.__func = func
        self.__name = name

    @property
    def name(self):
        return self.__name

    @property
    def func(self):
        return self.__func


class DerivableFunction(Function):
    """
    Class representing a function that we need the derivative of

    Attributes:
        deriv ( function "pointer"): Represents the derivative of the function
    """
    def __init__(self, func, deriv, name):
        super(DerivableFunction, self).__init__(func=func, name=name)
        self.__deriv = deriv

    @property
    def deriv(self):
        return self.__deriv

