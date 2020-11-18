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


def mean_squared_error(predicted, target):
    """
    Computes the mean squared error between
    the target vector and the output predicted by the net

    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth values for each of n examples
    :return: loss in terms of mse (Mean Squared Error)
    """
    # Check shapes
    if predicted.shape != target.shape:
        raise Exception(f"Mismatching shapes in MSE: predictions shape: "
                        f"{predicted.shape} - targets shape {target.shape}")

    #print(predicted - target)
    #print(np.square(predicted - target))
    #print(np.sum(np.square(predicted - target)))
    return np.sum(np.square(predicted - target)) / target.shape[0]


def mean_squared_error_deriv(predicted, target):
    """
    Computes the derivative of the mean squared error between
    the target vector and the output predicted by the net

    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth values for each of n examples
    :return: derivative of the mse (Mean Squared Error)
    """
    return predicted - target


def mean_euclidean_error(predicted, target):
    """
    Computes the Mean Euclidean Error between
    the target vector and the output predicted by the net

    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth values for each of n examples
    :return: loss in term of mee (Mean Euclidean Error)
    """
    return np.linalg.norm(predicted - target) / target.shape[0]


def mean_euclidean_error_deriv(predicted, target):
    """
    Computes the derivative of the Mean Euclidean Error between
    the target vector and the output predicted by the net

    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth values for each of n examples
    :return: derivative of the mee (Mean Euclidean Error)
    """
    return (predicted - target) / np.linalg.norm(predicted - target)


class Function:
    def __init__(self, func, deriv, name):
        self.func = func
        self.deriv = deriv
        self.name = name


# Objects that can be used many times just using their attributes (func, deriv)
Sigmoid = Function(sigmoid, sigmoid_deriv, 'Sigmoid')
ReLU = Function(relu, relu_deriv, 'ReLU')
MSE = Function(mean_squared_error, mean_squared_error_deriv, 'mse')
MEE = Function(mean_euclidean_error, mean_euclidean_error_deriv, 'mee')

functions = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    'mse': MSE,
    'mee': MEE
}

if __name__ == '__main__':
    # Test activation functions
    print("Activation functions test:")
    print(f"Sigmoid(1): {sigmoid(1)}")
    print(f"Derivative of sigmoid(1): {sigmoid_deriv(1)}")
    print(f"ReLU(1): {relu(1)}")
    print(f"ReLU(-3): {relu(-3)}")
    print(f"Derivative of ReLU(1): {relu_deriv(1)}")
    print(f"Derivative of ReLU(-3): {relu_deriv(-3)}")

    # Test loss functions
    print('\nArrays for testing loss functions:')
    y_true = np.array([[1, 1, 0, 0], [0, 0, 0, 0]])
    y_pred = np.array([[1, 0, 0, 1], [1, 1, 1, 1]])
    print(f"target : {y_true}")
    print(f"predicted: {y_pred}\n")
    print('Loss functions test:')
    print(f"MSE:{mean_squared_error(y_pred, y_true)}")
    print(f"MSE_deriv:{mean_squared_error_deriv(y_true, y_pred)}")
    print(f"MEE:{mean_euclidean_error(y_true, y_pred)}")
    print(f"MEE_deriv:{mean_euclidean_error_deriv(y_true, y_pred)}")
