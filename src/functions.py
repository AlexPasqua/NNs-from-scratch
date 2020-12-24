import numpy as np
from numbers import Number
import math


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
        deriv (function "pointer"): Represents the derivative of the function
    """

    def __init__(self, func, deriv, name):
        super(DerivableFunction, self).__init__(func=func, name=name)
        self.__deriv = deriv

    @property
    def deriv(self):
        return self.__deriv


def check_is_number(x):
    if isinstance(x, str):
        raise AttributeError(f"Input must be a number, got {type(x)}")
    if not isinstance(x, Number):
        if hasattr(x, '__iter__') and all(n == 1 for n in np.shape(x)):
            while hasattr(x, '__iter__'):
                x = x[0]
        if not isinstance(x, Number):
            raise AttributeError(f"Input must be a number. Got {type(x)}")


""" Activation Functions """


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


def leaky_relu(x):
    """
    Computes the leaky ReLu activation function
    :param x: input's weighted sum
    :return: leaky ReLu of x
    """
    check_is_number(x)
    return x if x >= 0 else 0.01 * x


def leaky_relu_deriv(x):
    """
    Computes the derivative of the leaky ReLu activation function
    :param x: input's weighted sum
    :return: derivative of the leaky ReLU in x
    """
    check_is_number(x)
    return 1 if x >= 0 else 0.01


""" Loss Functions """


def squared_loss(predicted, target):
    """
    Computes the mean squared error between
    the targ vector and the output pred by the net

    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth w_vals for each of n examples
    :return: loss in terms of mse (Mean Squared Error)
    """
    # Check shapes
    predicted = np.array(predicted)
    target = np.array(target)
    if predicted.shape != target.shape:
        raise AttributeError(f"Mismatching shapes in Squared Loss: predictions shape: "
                             f"{predicted.shape} - targets shape {target.shape}")
    return 0.5 * np.square(target - predicted)  # "0.5" is to make the gradient simpler


def squared_loss_deriv(predicted, target):
    """
    Computes the derivative of the mean squared error between
    the targ vector and the output pred by the net

    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth w_vals for each of n examples
    :return: derivative of the mse (Mean Squared Error)
    """
    predicted = np.array(predicted)
    target = np.array(target)
    if predicted.shape != target.shape:
        raise Exception(f"Mismatching shapes in MSE: predictions shape: "
                        f"{predicted.shape} - targets shape {target.shape}")

    # exponent 2 in the deriv becomes a multiplying constant and simplifies itself with the denominator of the func
    return predicted - target


""" Metrics """


def binary_class_accuracy(predicted, target):
    predicted = predicted[0]
    target = target[0]
    if np.abs(predicted - target) < 0.3:
        return np.array([1])
    return np.array([0])


BinClassAcc = Function(binary_class_accuracy, 'class_acc')
metrics = {
    'bin_class_acc': BinClassAcc
}

ReLU = DerivableFunction(relu, relu_deriv, 'ReLU')
LeakyReLU = DerivableFunction(leaky_relu, leaky_relu_deriv, 'LeakyReLU')
Sigmoid = DerivableFunction(sigmoid, sigmoid_deriv, 'Sigmoid')
Tanh = DerivableFunction(tanh, tanh_deriv, 'Tanh')
act_funcs = {
    'relu': ReLU,
    'leaky_relu': LeakyReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
}

SquaredLoss = DerivableFunction(squared_loss, squared_loss_deriv, 'squared')
losses = {
    'squared': SquaredLoss,
}

