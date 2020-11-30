import numpy as np


def lasso_l1(w, lambd):
    return lambd * np.sum(np.abs(w))


def ridge_l2(w, lambd):
    return lambd * np.sum(np.square(w))


def regularization(w, lambd, reg_type):
    """
    Computes the regularization
    :param w: weights vector
    :param lambd: regularization parameter (weight penalty)
    :param reg_type: type of regularization. "l2" = Ridge Regularization; "l1" = Lasso Regularization
    :return: the regularization factor
    """
    if lambd < 0:
        raise ValueError(f"'lambd' parameter must be >= 0")

    regularizations = {
        'l1': lasso_l1,
        'l2': ridge_l2
    }
    if reg_type not in regularizations:
        raise ValueError(f"Wrong regularization parameter: {reg_type} --> Chose among {list(regularizations.keys())}")

    return regularizations[reg_type](w, lambd)


if __name__ == '__main__':
    w = [1, 0.2, -1]
    print(f"Weights used for testing: {w}")
    print(f"L2 regularization(lamb = 0.2): {regularization(w, 0.2, 'l2')}")
    print(f"L1 regularization(lamb = 0.2): {regularization(w, 0.2, 'l1')}")
