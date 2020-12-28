import copy

import numpy as np
from network.layer import Layer
from functions import losses
from optimizers import *


class Network:
    """
    Neural network object
    Attributes:
        layers: list of net's layers ('Layer' objects)
    """

    def __init__(self, input_dim, units_per_layer, acts, init_type='uniform', init_value=0.2, **kwargs):
        """
        Constructor
        :param input_dim: the input dimension
        :param units_per_layer: list of layers' sizes as number on units
        :param acts: list of activation function names (one for each layer)
        """
        if not hasattr(units_per_layer, '__iter__'):
            units_per_layer = [units_per_layer]
            acts = [acts]
        self.__check_attributes(self,
                                input_dim=input_dim,
                                units_per_layer=units_per_layer,
                                acts=acts)

        self.__input_dim = input_dim
        self.__params = {**{
            'input_dim': input_dim,
            'units_per_layer': units_per_layer,
            'acts': acts,
            'init_type': init_type,
            'init_value': init_value
        }, **kwargs}
        self.__layers = []
        self.__opt = None
        other_args = {**{'init_type': init_type, 'init_value': init_value}, **kwargs}  # merge 2 dictionaries
        fanin = input_dim
        for i in range(len(units_per_layer)):
            self.__layers.append(Layer(fanin=fanin, n_units=units_per_layer[i], act=acts[i], **other_args))
            fanin = units_per_layer[i]

    @staticmethod
    def __check_attributes(self, input_dim, units_per_layer, acts):
        if input_dim < 1 or any(n_units < 1 for n_units in units_per_layer):
            raise ValueError("input_dim and every init_value in units_per_layer must be positive")
        if len(units_per_layer) != len(acts):
            raise AttributeError(
                f"Mismatching lengths --> len(units_per_layer)={len(units_per_layer)}; len(acts)={len(acts)}")

    @property
    def input_dim(self):
        return self.__input_dim

    @property
    def units_per_layer(self):
        return self.__params['units_per_layer']

    @property
    def layers(self):
        return self.__layers

    @property
    def opt(self):
        return self.__opt

    @property
    def params(self):
        return self.__params

    def forward(self, inp=(2, 2, 2), verbose=False):
        """
        Performs a prediction on the whole NN
        :param inp: net's input vector/matrix
        :return: net's output vector/matrix
        """
        if isinstance(inp, str):
            raise AttributeError("'inp' must be a vector of numbers, got string")
        inp = np.array(inp)
        # if inp is not iterable (e.g. single number)
        if len(inp.shape) == 0:
            inp = np.expand_dims(inp, 0)
        pattern_len = inp.shape[1] if len(inp.shape) > 1 else inp.shape[0]
        if pattern_len != self.__input_dim:
            raise AttributeError(f"Mismatching lengths --> len(net_inp) = {len(inp)} ; input_dim = {self.__input_dim}")
        if len(inp.shape) <= 1:
            inp = np.expand_dims(inp, 0)
        if verbose:
            print(f"Net's inputs: {inp}")

        outputs = []
        for pattern in inp:
            x = pattern
            for layer in self.layers:
                x = layer.forward_pass(x)
            if len(inp) > 1:
                outputs.append(x)
            else:
                outputs = x

        if verbose:
            print(f"Net's output: {outputs}")
        return outputs

    def compile(self, opt='gd', loss='squared', metr='bin_class_acc', lrn_rate=0.01, momentum=0.):
        """
        Prepares the network for training by assigning an optimizer to it
        :param opt: ('Optimizer' object)
        :param loss: (str) the type of loss function
        :param metr: (str) the type of metric to track (accuracy etc)
        :param lrn_rate: (float) learning rate value
        :param momentum: (float) momentum parameter
        """
        if opt not in optimizers or loss not in losses:
            raise AttributeError(f"opt must be within {optimizers.keys()} and loss must be in {losses.keys()}")
        if momentum > 1. or momentum < 0.:
            raise ValueError(f"momentum must be a value between 0 and 1. Got: {momentum}")
        self.__opt = optimizers[opt](net=self, loss=loss, metr=metr, lrn_rate=lrn_rate, momentum=momentum)


    def fit(self, tr_x, tr_y, val_x, val_y, epochs=1, batch_size=1):
        """
        Execute the training of the network
        :param tr_x: (numpy ndarray) input training set
        :param tr_y: (numpy ndarray) targets for each input training pattern
        :param val_x: (numpy ndarray) input validation set
        :param val_y: (numpy ndarray) targets for each input validation pattern
        :param batch_size: (integer) the size of the batch
        :param epochs: (integer) number of epochs
        """
        tr_y = np.array(tr_y)
        tr_x = np.array(tr_x)
        target_len = tr_y.shape[1] if len(tr_y.shape) > 1 else 1
        if target_len != len(self.layers[-1].units):
            raise AttributeError(
                f"Mismatching shapes --> target: {tr_y.shape} ; output units: {len(self.layers[-1].units)}")
        n_pattern = tr_x.shape[0] if len(tr_x.shape) > 1 else 1
        n_target = tr_y.shape[0] if len(tr_y.shape) > 1 else 1
        assert (n_pattern == n_target)
        return self.__opt.optimize(tr_x=tr_x, tr_y=tr_y, val_x=val_x, val_y=val_y, epochs=epochs, batch_size=batch_size)

    def propagate_back(self, dErr_dOut, grad_net):
        curr_delta = dErr_dOut
        for layer_index in reversed(range(len(self.__layers))):
            curr_delta, grad_w, grad_b = self.__layers[layer_index].backward_pass(curr_delta)
            grad_net[layer_index]['weights'] = np.add(grad_net[layer_index]['weights'], grad_w)
            grad_net[layer_index]['biases'] = np.add(grad_net[layer_index]['biases'], grad_b)
        return grad_net

    def get_empty_struct(self):
        """
        :return: a zeroed structure to contain all the layers' gradients
        """
        struct = np.array([{}] * len(self.__layers))
        for layer_index in range(len(self.__layers)):
            struct[layer_index] = {'weights': [], 'biases': []}
            struct[layer_index]['weights'] = np.array([0.] * len(self.__layers[layer_index].weights))
            struct[layer_index]['biases'] = np.array([0.] * len(self.__layers[layer_index].biases))
        return struct

    def print_net(self):
        """ Prints the network's architecture and parameters """
        print('Neural Network:')
        for layer in self.layers:
            print('  Layer:')
            for unit in layer.units:
                print(f"\tUnit:\n\t  weights: {unit.w}\n\t  bias: {unit.b}\n\t  activation function: {unit.act.name}")
            print()
