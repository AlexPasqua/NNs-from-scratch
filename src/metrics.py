import numpy as np
from Function import Function


def mean_euclidean_error(predicted, target):
    """
    Computes the Mean Euclidean Error between
    the targ vector and the output pred by the net over all patterns

    :type predicted: object
    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth w_vals for each of n examples
    :return: loss in term of mee (Mean Euclidean Error)
    """
    if predicted.shape != target.shape:
        raise AttributeError(f"Mismatching shapes in MEE: predictions shape: "
                             f"{predicted.shape} - targets shape {target.shape}")
    n_patterns = predicted.shape[0]
    return np.linalg.norm(predicted - target) / n_patterns


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


def classification_accuracy():
    pass


MEE = Function(mean_euclidean_error, 'mee')
metrics = {
    'mee': MEE,
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

    print(f"MEE:{metrics['mee'].func(predicted, target)}")
