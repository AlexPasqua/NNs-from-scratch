import numpy as np
from src.Function import DerivableFunction


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

    # MSE of the whole net would be
    # np.sum(np.square(pred - targ)) / (targ.shape[0] * 2)
    return 0.5 * np.square(predicted - target)    # "0.5" is to make the gradient simpler


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
    pred = [1, 0, 0, 1]
    targ = [1, 1, 0, 0]

    print(f"pred:\t{pred}\n")
    print(f"targ:\t{targ}")
    print('Loss functions test:')
    print(f"squared loss:\t{losses['squared'].func(pred, targ)}")
    print(f"squared loss deriv:\t{losses['squared'].deriv(pred, targ)}")

