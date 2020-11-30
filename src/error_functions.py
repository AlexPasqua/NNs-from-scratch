import numpy as np
from Function import Function


def mean_euclidean_error(predicted, target):
    """
    Computes the Mean Euclidean Error between
    the target vector and the output predicted by the net

    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth values for each of n examples
    :return: loss in term of mee (Mean Euclidean Error)
    """
    if predicted.shape != target.shape:
        raise Exception(f"Mismatching shapes in MSE: predictions shape: "
                        f"{predicted.shape} - targets shape {target.shape}")
    return np.linalg.norm(predicted - target) / target.shape[0]


# def mean_euclidean_error_deriv(predicted, target):
#     """
#     Computes the derivative of the Mean Euclidean Error between
#     the target vector and the output predicted by the net
#
#     :param predicted: ndarray of shape (n, m) – Predictions for the n examples
#     :param target: ndarray of shape (n, m) – Ground truth values for each of n examples
#     :return: derivative of the mee (Mean Euclidean Error)
#     """
#     if predicted.shape != target.shape:
#         raise Exception(f"Mismatching shapes in MSE: predictions shape: "
#                         f"{predicted.shape} - targets shape {target.shape}")
#     return np.sum(predicted - target) / (target.shape[0] * np.linalg.norm(predicted - target))


MEE = Function(mean_euclidean_error, None, 'mee')
err_funcs = {
    'mee': MEE
}

if __name__ == '__main__':
    predicted = np.array(
        [[1, 0, 0, 1],
         [1, 1, 1, 1]]
    )
    target = np.array(
        [[1, 1, 0, 0],
         [0, 0, 0, 0]]
    )

    print(f"MEE:{err_funcs['mee'].func(predicted, target)}")
