import numpy as np
from Function import DerivableFunction
from numbers import Number
import math


def check_is_number(x):
    if not isinstance(x, Number):
        raise AttributeError(f"Input of sigmoid must be a number. Got {type(x)}")


def sigmoid(x):
    """
    Computes the sigmoid function of x

    :param x: net -> input's weighted sum
    :return: sigmoid of x
    """
    check_is_number(x)
    return 1. / (1. + np.exp(-x))


def sigmoid_deriv(x):
    """
    Computes the derivative of the sigmoid function

    :param x: net -> input's weighted sum
    :return: derivative of the sigmoid in x
    """
    check_is_number(x)
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    """
    Computes the ReLU function:

    :param x: net -> input's weighted sum
    :return: ReLU of x
    """
    check_is_number(x)
    return np.maximum(0, x)


def relu_deriv(x):
    """
    Computes the derivative of the ReLU function:

    :param x: net-> input's weighted sum
    :return: derivative of the ReLU in x
    """
    check_is_number(x)
    return 0 if x <= 0 else 1


def tanh(x):
    """
    Computes the hyperbolic tangent function (tanh) of x
    :param x: net-> input's weighted sum
    :return: Tanh of x
    """
    check_is_number(x)
    return math.tanh(x)


def tanh_deriv(x):
    """
    Computes the derivative of the hyperbolic tangent function (tanh)
    :param x: net-> input's weighted sum
    :return: Tanh derivative of x
    """
    check_is_number(x)
    return 1 - (math.tanh(x)) ** 2


# Objects that can be used many times just using their attributes (func, deriv)
Sigmoid = DerivableFunction(sigmoid, sigmoid_deriv, 'Sigmoid')
ReLU = DerivableFunction(relu, relu_deriv, 'ReLU')
Tanh = DerivableFunction(tanh, tanh_deriv, 'Tanh')

act_funcs = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh
}

if __name__ == '__main__':
    # Test activation functions
    print("Activation functions test:")
    print(f"Sigmoid(1): {act_funcs['sigmoid'].func(1)}")
    print(f"Derivative of sigmoid(1): {act_funcs['sigmoid'].deriv(1)}")
    print(f"ReLU(1): {act_funcs['relu'].func(1)}")
    print(f"ReLU(-3): {act_funcs['relu'].func(-3)}")
    print(f"Derivative of ReLU(1): {act_funcs['relu'].deriv(1)}")
    print(f"Derivative of ReLU(-3): {act_funcs['relu'].deriv(-3)}")
    print(f"Tanh(1): {act_funcs['tanh'].func(1)}")
    print(f"Derivative of tanh(1): {act_funcs['tanh'].deriv(1)}")
