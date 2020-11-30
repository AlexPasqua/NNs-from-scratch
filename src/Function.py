""" Script containing the definition of the class representing a function"""


class Function:
    """
    Class representing a function

    Attributes:
        func: function "pointer"
            Represents the function itself

        name: string
            name of the function
    """
    def __init__(self, func, name):
        self.func = func
        self.name = name

    def get_name(self):
        return self.name


class DerivableFunction(Function):
    """
    Class representing a function that we need the derivative of

    Attributes:
        deriv: function "pointer"
            Represents the derivative of the function
    """
    def __init__(self, func, deriv, name):
        super(DerivableFunction, self).__init__(func=func, name=name)
        self.deriv = deriv

