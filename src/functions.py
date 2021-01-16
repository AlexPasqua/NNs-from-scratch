import math

import numpy as np


class Function:
    """
    Class representing a function
    Attributes:
        func (function "pointer"): Represents the function itself
        name (string): name of the function
    """

    def __init__(self, func, name):
        self.__func = func
        self.__name = name

    @property
    def name(self):
        return self.__name

    @property
    def func(self):
        return self.__func


class DerivableFunction(Function):
    """
    Class representing a function that we need the derivative of
    Attributes:
        deriv (function "pointer"): Represents the derivative of the function
    """

    def __init__(self, func, deriv, name):
        super(DerivableFunction, self).__init__(func=func, name=name)
        self.__deriv = deriv

    @property
    def deriv(self):
        return self.__deriv


""" Activation Functions """


def identity(x):
    """
    Computes the identity function
    :param x:  net -> input's weighted sum
    :return: x
    """
    return x


def indentity_deriv(x):
    """
    Computes the derivative of the identity function
    :param x: net -> input's weighted sum
    :return: derivative of identity of x (i.e. 1)
    """
    return 1.


def relu(x):
    """
    Computes the ReLU function:
    :param x: net -> input's weighted sum
    :return: ReLU of x
    """
    return np.maximum(x, 0)


def relu_deriv(x):
    """
    Computes the derivative of the ReLU function:
    :param x: net-> input's weighted sum
    :return: derivative of the ReLU in x
    """
    x = np.array(x)
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def leaky_relu(x):
    """
    Computes the leaky ReLu activation function
    :param x: input's weighted sum
    :return: leaky ReLu of x
    """
    return [i if i >= 0 else 0.01 * i for i in x]


def leaky_relu_deriv(x):
    """
    Computes the derivative of the leaky ReLu activation function
    :param x: input's weighted sum
    :return: derivative of the leaky ReLU in x
    """
    x = np.array(x)
    x[x > 0] = 1.
    x[x <= 0] = 0.01
    return x


def sigmoid(x):
    """
    Computes the sigmoid function of x
    :param x: net -> input's weighted sum
    :return: sigmoid of x
    """
    x = np.array(x)
    ones = [1.] * len(x)
    return np.divide(ones, np.add(ones, np.exp(-x)))


def sigmoid_deriv(x):
    """
    Computes the derivative of the sigmoid function
    :param x: net -> input's weighted sum
    :return: derivative of the sigmoid in x
    """
    return np.multiply(
        sigmoid(x),
        np.subtract([1.] * len(x), sigmoid(x))
    )


def tanh(x):
    """
    Computes the hyperbolic tangent function (tanh) of x
    :param x: net-> input's weighted sum
    :return: Tanh of x
    """
    return np.tanh(x)


def tanh_deriv(x):
    """
    Computes the derivative of the hyperbolic tangent function (tanh)
    :param x: net-> input's weighted sum
    :return: Tanh derivative of x
    """
    return np.subtract(
        [1.] * len(x),
        np.power(np.tanh(x), 2)
    )


""" Loss Functions """


def squared_loss(predicted, target):
    """
    Computes the squared error between the targ vector and the output pred by the net
    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth w_vals for each of n examples
    :return: loss in terms of squared error
    """
    return 0.5 * np.square(np.subtract(target, predicted))  # "0.5" is to make the gradient simpler


def squared_loss_deriv(predicted, target):
    """
    Computes the derivative of the squared error between the targ vector and the output pred by the net
    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth w_vals for each of n examples
    :return: derivative of the squared error
    """
    # exponent 2 in the deriv becomes a multiplying constant and simplifies itself with the denominator of the func
    return np.subtract(predicted, target)


""" Metrics """


def binary_class_accuracy(predicted, target):
    predicted = predicted[0]
    target = target[0]
    if np.abs(predicted - target) < 0.3:
        return np.array([1])
    return np.array([0])


def euclidean_loss(predicted, target):
    """
    Computes the euclidean error between the targ vector and the output pred by the net
    :param predicted: ndarray of shape (n, m) – Predictions for the n examples
    :param target: ndarray of shape (n, m) – Ground truth w_vals for each of n examples
    :return: loss in terms of euclidean error
    """
    return np.linalg.norm(np.subtract(predicted, target))


""" Learning rate decay """


def linear_lr_decay(curr_lr, base_lr, final_lr, curr_step, limit_step):
    """
    The linear_lr_decay, linearly decays the learning rate until iteration tau (limit_step). Then it stops
    decaying and uses a fix learning rate (final_lr)::

    :param curr_lr:
    :param base_lr:
    :param final_lr:
    :param curr_step:
    :param limit_step:
    :return:
    """
    if curr_step < limit_step and curr_lr > final_lr:
        decay_rate = curr_step / limit_step
        curr_lr = (1. - decay_rate) * base_lr + decay_rate * final_lr
        return curr_lr
    return final_lr


def exp_lr_decay(initial_lr, decay_rate, step, decay_steps, staircase=True):
    """
    The exp_lr_decay, decays exponentially the learning rate by `decay_rate` every
        `decay_step`, starting from `initial_lr`::
            learning_rate = initial_lr * exp(-decay_rate * cur_stage)
        where::
            cur_stage = step / decay_steps          if staircase = False
            cur_stage = floor(step / decay_steps)   if staircase = True

    :param initial_lr: The learning rate at the first step
    :param decay_rate: The amount to decay the learning rate at each new stage
    :param step: corresponds to the epoch num
    :param decay_steps: The length of each stage, in steps
    :param staircase: If True, only adjusts the learning rate at the stage transitions,
                      producing a step-like decay schedule.
                      If False, adjusts the learning rate after each step,
                      creating a smooth decay schedule.
                      Default is True
    :return: exponentially decayed learning rate
    """
    cur_stage = step/decay_steps
    if staircase:
        cur_stage = np.floor(cur_stage)
    decay = -decay_rate * cur_stage
    return initial_lr * math.exp(decay)


""" Regularizations """


def lasso_l1(w, lambd):
    return lambd * np.sum(np.abs(w))


def lasso_l1_deriv(w, lambd):
    res = np.zeros(len(w))
    for i in range(len(w)):
        if w[i] < 0:
            res[i] = -lambd
        elif w[i] > 0:
            res[i] = lambd
        else:
            res[i] = 0
    return res


def ridge_l2(w, lambd):
    return lambd * np.sum(np.square(w))


def ridge_l2_deriv(w, lambd):
    return 2 * lambd * w


""" Function objects and dictionaries to use them in other scripts """


Identity = DerivableFunction(identity, indentity_deriv, 'identity')
ReLU = DerivableFunction(relu, relu_deriv, 'ReLU')
LeakyReLU = DerivableFunction(leaky_relu, leaky_relu_deriv, 'LeakyReLU')
Sigmoid = DerivableFunction(sigmoid, sigmoid_deriv, 'Sigmoid')
Tanh = DerivableFunction(tanh, tanh_deriv, 'Tanh')
act_funcs = {
    'identity': Identity,
    'relu': ReLU,
    'leaky_relu': LeakyReLU,
    'sigmoid': Sigmoid,
    'tanh': Tanh,
}

SquaredLoss = DerivableFunction(squared_loss, squared_loss_deriv, 'squared')
losses = {
    'squared': SquaredLoss,
}

BinClassAcc = Function(binary_class_accuracy, 'class_acc')
Euclidean = Function(euclidean_loss, 'euclidean')
metrics = {
    'bin_class_acc': BinClassAcc,
    'euclidean': Euclidean
}

LinearLRDecay = Function(linear_lr_decay, 'linear')
ExponentialLRDecay = Function(exp_lr_decay, 'exponential')

lr_decays = {
    'linear': LinearLRDecay,
    'exponential': ExponentialLRDecay
}

l2_regularization = DerivableFunction(ridge_l2, ridge_l2_deriv, 'l2')
l1_regularization = DerivableFunction(lasso_l1, lasso_l1_deriv, 'l1')
regs = {
    'l2': l2_regularization,
    'l1': l1_regularization
}