import numpy as np
from Function import DerivableFunction


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


SquaredLoss = DerivableFunction(squared_loss, squared_loss_deriv, 'squared')
losses = {
    'squared': SquaredLoss,
}

if __name__ == '__main__':
    # Test loss functions
    print('\nArrays for testing loss functions:')
    pred = [[1, 0, 0, 1], [1, 0, 1, 1]]
    targ = [[1, 0, 0, 0], [1, 1, 0, 1]]

    print(f"pred:\t{pred}")
    print(f"targ:\t{targ}\n")
    print('Loss functions test:')
    print(f"squared loss:\n{losses['squared'].func(pred, targ)}\n")
    print(f"squared loss deriv:\n{losses['squared'].deriv(pred, targ)}")
