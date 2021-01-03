import numpy as np
from weights_initializations import weights_inits
from functions import act_funcs


class Layer:
    """
    Class that represent a layer of a neural network
    Attributes:
    """

    def __init__(self, inp_dim, n_units, act, init_type, **kwargs):
        """
        :param n_units: (integer) number of units in the layer
        :param init_type: (string) type of weights initialization
        """
        self.weights = weights_inits(init_type=init_type, n_weights=inp_dim, n_units=n_units, **kwargs)
        self.biases = weights_inits(init_type=init_type, n_weights=1, n_units=n_units, **kwargs)
        self.__inp_dim = inp_dim
        self.__n_units = n_units
        self.__act = act_funcs[act]
        self.__inputs = None
        self.__nets = None
        self.__outputs = None
        self.__gradient_w = None
        self.__gradient_b = None

    @property
    def inp_dim(self):
        return self.__inp_dim

    @property
    def act(self):
        return self.__act

    @property
    def n_units(self):
        return self.__n_units

    @property
    def inputs(self):
        return self.__inputs

    @property
    def nets(self):
        return self.__nets

    @property
    def outputs(self):
        return self.__outputs

    def forward_pass(self, inp):
        """
        Performs the forward pass on the current layer
        :param inp: (numpy ndarray) input vector
        :return: the vector of the current layer's soutputs
        """
        self.__inputs = np.array(inp)
        self.__nets = np.matmul(inp, self.weights)
        self.__nets = np.add(self.__nets, self.biases)
        self.__outputs = self.__act.func(self.__nets)
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
        dOut_dNet = self.__act.deriv(self.__nets)
        delta = np.multiply(upstream_delta, dOut_dNet)
        self.__gradient_b = -delta
        self.__gradient_w = np.zeros(shape=(self.__inp_dim, self.__n_units))
        for i in range(self.__inp_dim):
            for j in range(self.__n_units):
                self.__gradient_w[i][j] = -delta[j] * self.__inputs[i]
        # the i-th row of the weights matrix corresponds to the vector formed by the i-th weight of each layer's unit
        new_upstream_delta = [np.dot(delta, self.weights[i]) for i in range(self.__inp_dim)]
        return new_upstream_delta, self.__gradient_w, self.__gradient_b
