import numpy as np
from numbers import Number
from network.unit import Unit


class Layer:
    """
    Class that represent a layer of a neural network
    Attributes:
    """

    def __init__(self, fanin=None, n_units=None, act=None, init_type=None, units=None, **kwargs):
        """
        :param n_units: (integer) number of units in the layer
        :param init_type: (string) type of weights initialization
        """
        self.__inputs = None
        self.__outputs = None
        self.__gradient_w = None
        self.__gradient_b = None
        # if units is empty
        if units is None:
            args = {'fanin': fanin, 'n_units': n_units, 'act': act, 'init_type': init_type}
            if any(arg is None for arg in args.values()):
                raise AttributeError(f"If a list of Unit(s) is not provided, every one in {list(args.keys())} must be initialized")
            self.__units = []
            unit_params = {**{'init_type': init_type, 'act': act, 'n_weights': fanin}, **kwargs}
            for i in range(n_units):
                self.__units.append(Unit(**unit_params))
        else:
            self.__units = units
            if np.shape(self.__units) == 0:
                self.__units = np.expand_dims(self.__units, 0)

    @property
    def units(self):
        return self.__units

    @property
    def weights(self):
        return np.array([u.w[i] for u in self.__units for i in range(len(u.w))])

    @property
    def biases(self):
        return np.array([u.b for u in self.__units])

    @property
    def outputs(self):
        return self.__outputs

    @property
    def act(self):
        return self.units[0].act

    @staticmethod
    def __check_vectors(self, passed, own):
        if hasattr(passed, '__iter__'):
            if not all(isinstance(n, Number) for n in passed):
                raise ValueError("layer's weights must be numeric. Got: ", type(passed[0]))
            if len(passed) != len(own):
                raise AttributeError("'value' must have the same length of the layer's weights")
        else:
            raise AttributeError(f"'value' must be a iterable, got {type(passed)}")

    @weights.setter
    def weights(self, value):
        self.__check_vectors(self, passed=value, own=self.weights)
        for i in range(len(self.units)):
            n_weights = len(self.units[i].w)
            start = i * n_weights
            end = start + n_weights
            self.units[i].w = value[start: end]

    @biases.setter
    def biases(self, value):
        self.__check_vectors(self, passed=value, own=self.biases)
        for i in range(len(self.units)):
            self.units[i].b = value[i]

    def forward_pass(self, inp):
        """
        Performs the forward pass on the current layer
        :param inp: (numpy ndarray) input vector
        :return: the vector of the current layer's soutputs
        """
        self.__inputs = inp
        self.__outputs = [unit.output(inp) for unit in self.units]
        return self.__outputs

    def backward_pass(self, upstream_delta):
        """
        Sets the layer's gradients
        :param upstream_delta: for hidden layers, delta = dot_prod(delta_next, w_next) * dOut_dNet
            Multiply (dot product) already the delta for the current layer's weights in order to have it ready for the previous
            layer (that does not have access to this layer's weights), that will execute this method in the
            next iteration of Network.propagate_back()
        :return new_upstream_delta: delta already multiplied (dot product) by the current layer's weights
        :return gradient_w: gradient wrt weights
        :return gradient_b: gradient wrt biases
        """
        dOut_dNet = np.array([self.act.deriv(u.net) for u in self.__units])
        delta = upstream_delta * dOut_dNet
        self.__gradient_b = -delta
        self.__gradient_w = [
            -delta[j] * self.__inputs[i]
            for j in range(len(delta))
            for i in range(len(self.__inputs))
        ]
        new_upstream_delta = [np.dot(delta, [u.w[j] for u in self.units]) for j in range(len(self.__inputs))]
        return new_upstream_delta, self.__gradient_w, self.__gradient_b
