import numpy as np


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
    return np.diag(sigmoid(x) * (1 - sigmoid(x)))


def softmax(x):
    """
    Computes the softmax function of the input received by the unit:

    :param x: net -> input's weighted sum
    :return: softmax of x
    """
    return np.exp(x - np.max(x))


def softmax_deriv(x): #TODO: to be completed
    """
    Computes the derivative of the softmax function:

    :param x: net -> input's weighted sum
    :return: derivative of the softmax in x
    """
    softmax_fun = np.exp(x - np.max(x))


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
    return 0 if x < 0 else 1


if __name__ == '__main__':
    # TODO: aggiungi test qui
    print(sigmoid(1))
