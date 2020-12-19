import numpy as np
from weights_initializations import weights_inits
from functions import act_funcs


class Unit:
    """
    Class that represent a neuron
    Attributes:
        w (list of float): weights
        b (float): bias
    """
    def __init__(self, init_type, act, **kwargs):
        """
        :param init_type: (string) type of weights initialization
        """
        kwargs['type'] = init_type
        self.w = weights_inits(**kwargs)
        kwargs['n_weights'] = 1
        self.b = weights_inits(**kwargs)
        self.act = act_funcs[act]
        self.__net = None
        self.__out = None

    @property
    def net(self):
        return self.__net

    def output(self, inp):
        """
        Computes activation function on the weighted sum of the input
        :param inp: unit's input vector
        :return: unit's output
        """
        # compute activation function on weighted sum
        self.__net = np.dot(inp, self.w) + self.b
        self.__out = self.act.func(self.__net)
        return self.__out
