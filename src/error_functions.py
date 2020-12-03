import numpy as np
from Function import Function


def mean_euclidean_error(predicted, target):
    """
    Computes the Mean Euclidean Error between
    the targ vector and the output pred by the net

    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth w_vals for each of n examples
    :return: loss in term of mee (Mean Euclidean Error)
    """
    if predicted.shape != target.shape:
        raise Exception(f"Mismatching shapes in MSE: predictions shape: "
                        f"{predicted.shape} - targets shape {target.shape}")
    return np.linalg.norm(predicted - target) / target.shape[0]


# def mean_euclidean_error_deriv(pred, targ):
#     """
#     Computes the derivative of the Mean Euclidean Error between
#     the targ vector and the output pred by the net
#
#     :param pred: ndarray of shape (n, m) – Predictions for the n examples
#     :param targ: ndarray of shape (n, m) – Ground truth w_vals for each of n examples
#     :return: derivative of the mee (Mean Euclidean Error)
#     """
#     if pred.shape != targ.shape:
#         raise Exception(f"Mismatching shapes in MSE: predictions shape: "
#                         f"{pred.shape} - targets shape {targ.shape}")
#     return np.sum(pred - targ) / (targ.shape[0] * np.linalg.norm(pred - targ))


MEE = Function(mean_euclidean_error, 'mee')
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
