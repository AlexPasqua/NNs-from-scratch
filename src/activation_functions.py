import numpy as np
from Function import DerivableFunction


def sigmoid(x):
    """
    Computes the sigmoid function of x

    :param x: net -> input's weighted sum
    :return: sigmoid of x
    """
    return 1. / (1. + np.exp(-x))


def sigmoid_deriv(x):
    """
    Computes the derivative of the sigmoid function

    :param x: net -> input's weighted sum
    :return: derivative of the sigmoid in x
    """
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    """
    Computes the ReLU function:

    :param x: net -> input's weighted sum
    :return: ReLU of x
    """
    return np.maximum(0, x)


def relu_deriv(x):
    """
    Computes the derivative of the ReLU function:

    :param x: net-> input's weighted sum
    :return: derivative of the ReLU in x
    """
    return 0 if x <= 0 else 1


# Objects that can be used many times just using their attributes (func, deriv)
Sigmoid = DerivableFunction(sigmoid, sigmoid_deriv, 'Sigmoid')
ReLU = DerivableFunction(relu, relu_deriv, 'ReLU')

act_funcs = {
    'relu': ReLU,
    'sigmoid': Sigmoid
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
