""" Script containing the definition of the class representing a function"""


class Function:
    """
    Class representing a function

    Attributes:
        func: function "pointer"
            Represents the function itself

        deriv: function "pointer"
            Represents the derivative of the function

        name: string
            name of the function
    """

    def __init__(self, func, deriv, name):
        self.func = func
        self.deriv = deriv
        self.name = name
