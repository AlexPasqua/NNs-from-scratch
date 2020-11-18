import numpy as np


def regularization(w, lambd, l_n):
    """
    Computes the regularization
    :param w: weights vector
    :param lambd: regularization parameter (weight penalty)
    :param l_n: type of regularization. "l2" = Ridge Regularization; "l1" = Lasso Regularization
    :return: the regularization factor
    """

    if l_n == 'l2':
        return lambd * np.sum(np.square(w))
    elif l_n == 'l1':
        return lambd * np.sum(np.abs(w))
    else:
        raise ValueError("l_n should be either l1 or l2")


if __name__ == '__main__':
    w = ([1, 0.2, -1])
    print(f"Weights used for testing: {w}")
    print(f"L2 regularization(lamb = 0.2): {regularization(w, 0.2, 'l2')}")
    print(f"L1 regularization(lamb = 0.2): {regularization(w, 0.2, 'l1')}")
