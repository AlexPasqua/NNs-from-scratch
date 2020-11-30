import numpy as np
from Function import DerivableFunction


def squared_loss(predicted, target):
    """
    Computes the mean squared error between
    the target vector and the output predicted by the net

    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth values for each of n examples
    :return: loss in terms of mse (Mean Squared Error)
    """
    # Check shapes
    predicted = np.array(predicted)
    target = np.array(target)
    if predicted.shape != target.shape:
        raise Exception(f"Mismatching shapes in MSE: predictions shape: "
                        f"{predicted.shape} - targets shape {target.shape}")

    # MSE of the whole net would be
    # np.sum(np.square(predicted - target)) / (target.shape[0] * 2)
    return 0.5 * np.sum(np.square(predicted - target), axis=0)    # "0.5" is to make the gradient simpler


def squared_loss_deriv(predicted, target):
    """
    Computes the derivative of the mean squared error between
    the target vector and the output predicted by the net

    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth values for each of n examples
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
    predicted = np.array(
        [[1, 0, 0, 1],
         [1, 1, 1, 1]]
    )
    target = np.array(
        [[1, 1, 0, 0],
         [0, 0, 0, 0]]
    )

    print(f"target:\n{target}")
    print(f"predicted:\n{predicted}\n")
    print('Loss functions test:')
    print(f"squared loss:{losses['squared'].func(predicted, target)}")
    print(f"squared loss deriv:{losses['squared'].deriv(predicted, target)}")

