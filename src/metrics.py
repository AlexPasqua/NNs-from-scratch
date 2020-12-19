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


def binary_class_accuracy(predicted, target):
    # lower_thresh = 0.4
    # upper_thresh = 0.6
    predicted = predicted[0]
    target = target[0]
    # if predicted < lower_thresh or predicted > upper_thresh:
    #     if abs(predicted - target) < lower_thresh:
    #         return [1]
    if np.abs(predicted - target) < 0.2:
        return np.array([1])
    return np.array([0])


MEE = Function(mean_euclidean_error, 'mee')
ClassAcc = Function(binary_class_accuracy, 'class_acc')
metrics = {
    'mee': MEE,
    'class_acc': ClassAcc
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
