import numpy as np
from functions import DerivableFunction


def lasso_l1(w, lambd):
    return lambd * np.sum(np.abs(w))


def lasso_l1_deriv(w, lambd):
    for i in range((w.shape[0])):
        if w[i] < 0:
            w[i] = -lambd
        else:
            w[i] = lambd
    return w


def ridge_l2(w, lambd):
    return lambd * np.sum(np.square(w))


def ridge_l2_deriv(w, lambd):
    return 2 * lambd * w


'''
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
'''
l2_regularization = DerivableFunction(ridge_l2, ridge_l2_deriv, 'l2')
l1_regularization = DerivableFunction(lasso_l1, lasso_l1_deriv, 'l1')
regularization = {
    'l2': l2_regularization,
    'l1': l1_regularization
}

if __name__ == '__main__':
    w = np.array([1, 0.2, -1])
    print(f"Weights used for testing: {w}")
    print(f"L2 regularization func (lambd = 0.2):{regularization['l2'].func(w=w, lambd=0.2)}")
    print(f"L2 regularization deriv (lambd = 0.2):{regularization['l2'].deriv(w=w, lambd=0.2)}")
    print(f"L1 regularization func (lambd = 0.2):{regularization['l1'].func(w=w, lambd=0.2)}")
    print(f"L1 regularization deriv (lambd = 0.2):{regularization['l1'].deriv(w=w, lambd=0.2)}")
