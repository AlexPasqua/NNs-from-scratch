import copy
from numbers import Number

import numpy as np
import argparse
from activation_functions import act_funcs
from optimizers import *
from losses import losses
from numbers import Number
from weights_inits import weights_inits


class Unit:
    """
    This class represent a unit/neuron of a NN

    Attributes:
        w: weights vector
        b: bias
        act: activation function
    """

    def __init__(self, w, b, act):
        """
        Constructor
        :param w: vector of weights
        :param b: bias (number)
        :param act: activation function --> 'Function' obj (see 'functions.py')
        """
        self.__w = w
        self.__b = b
        self.__act = act
        self.__net = None
        self.__out = None
        self.__grad_w = [0.] * len(self.__w)
        self.__grad_b = 0.

    @property
    def w(self):
        return self.__w

    @property
    def b(self):
        return self.__b

    @property
    def act(self):
        return self.__act

    @property
    def out(self):
        return self.__out

    @property
    def net(self):
        return self.__net
    
    @property
    def grad_w(self):
        return self.__grad_w
    
    @property
    def grad_b(self):
        return self.__grad_b

    @w.setter
    def w(self, value):
        self.__w = value

    @b.setter
    def b(self, value):
        self.__b = value

    @grad_w.setter
    def grad_w(self, value):
        self.__grad_w = value
    
    @grad_b.setter
    def grad_b(self, value):
        self.__grad_b = value

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


class Layer:
    """
    This class implements a layer of a NN

    Attributes:
        units: list of layer's units ('Unit' objects)
    """
    def __init__(self, units):
        """
        Constructor
        :param units: list on layer's units ('Unit' objects)
        """
        units_acts = [u.act for u in units]
        for act in units_acts[1:]:
            if act != units_acts[0]:
                raise ValueError("All units in a layer must have the same activation function")
        self.__units = units
        self.__act = self.__units[0].act
        self.__outputs = []
        self.__inputs = []

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
    def gradient_w(self):
        return np.array([u.grad_w[i] for u in self.__units for i in range(len(u.grad_w))])

    @property
    def gradient_b(self):
        return np.array([u.grad_b for u in self.__units])

    # @property
    # def weights_biases(self):
    #     """
    #     :return: vector of layer's weights and biases
    #     """
    #     wb = np.zeros([len(self.units) * (len(self.units[0].w) + 1)])
    #     for j in range(len(self.units)):
    #         u = self.units[j]
    #         offset = len(u.w) + 1
    #         wb[offset * (j + 1) - 1] = u.b
    #         for k in range(len(u.w)):
    #             wb[k + offset * j] = u.w[k]
    #     return wb

    @staticmethod
    def __check_vectors(self, passed, own):
        if hasattr(passed, '__iter__'):
            if not all(isinstance(n, Number) for n in passed):
                raise ValueError("layer's weights must be numeric. Got: ", type(passed[0]))
            if len(passed) != len(own):
                raise AttributeError("'value' must have the same length of the layer's weights")
        else:
            raise AttributeError(f"'value' must be a iterable, got {type(passed)}")

    @property
    def act(self):
        return self.__act

    @property
    def outputs(self):
        return self.__outputs

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

    # @weights_biases.setter
    # def weights_biases(self, value):
    #     self.__check_vectors(self, passed=value, own=self.weights_biases)
    #     for i in range(len(self.units)):
    #         n = len(self.units[i].w)
    #         start = i * n
    #         end = start + n
    #         self.units[i].w = value[start: end]
    #         self.units[i].b = value[end]

    @gradient_w.setter
    def gradient_w(self, value):
        for i in range(len(self.__units)):
            n_weights = len(self.__units[i].w)
            start = i * n_weights
            end = start + n_weights
            self.__units[i].grad_w = value[start: end]

    @gradient_b.setter
    def gradient_b(self, value):
        self.__check_vectors(self, passed=value, own=self.biases)
        for i in range(len(self.__units)):
            self.__units[i].grad_b = value[i]

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
        Sets the layer's gradient
        """
        dOut_dNet = np.array([self.__act.deriv(u.net) for u in self.__units])
        delta = upstream_delta * dOut_dNet
        self.gradient_b = -delta
        self.gradient_w = [
            -delta[j] * self.__inputs[i]
            for j in range(len(delta))
            for i in range(len(self.__inputs))
        ]
        new_upstream_delta = [np.dot(delta, [u.w[j] for u in self.__units]) for j in range(len(self.__inputs))]
        return new_upstream_delta


class Network:
    """
    This class creates NNs

    Attributes:
        layers: list of net's layers ('Layer' objects)
    """
    def __init__(self, input_dim=3, units_per_layer=(3, 2), acts=('relu', 'sigmoid'), weights_init='uniform',
                 weights_value=0.1, **kwargs):
        """
        Constructor
        :param input_dim: the input dimension
        :param units_per_layer: list of layers' sizes as number on units
        :param acts: list of activation function names (one for each layer)
        """
        self.__check_attributes(self,
                                input_dim=input_dim,
                                units_per_layer=units_per_layer,
                                acts=acts,
                                weights_init=weights_init,
                                weights_value=weights_value)

        self.__input_dim = input_dim
        self.__units_per_layer = units_per_layer
        self.__acts = acts
        self.__layers = []
        self.__opt = None
        units = []

        # for each layer...
        for i in range(len(units_per_layer)):
            # number of weights of the units in a certain layer
            n_weights = input_dim if i == 0 else len(self.layers[i - 1].units)

            # for every unit in the current layer create layer's units
            for j in range(units_per_layer[i]):
                #TODO: this part needs to be checked. It overrites the values assigned by weight_init
                units.append(
                    Unit(w=weights_inits(type=weights_init, n_weights=n_weights, lower_lim=0.1, upper_lim=0.2,
                                         value=weights_value),
                         b=weights_inits(type=weights_init, n_weights=1, lower_lim=0., upper_lim=0.01,
                                         value=weights_value),
                         act=act_funcs[acts[i]])
                )
            self.layers.append(Layer(units=units))
            units = []

    @staticmethod
    def __check_attributes(self, input_dim, units_per_layer, acts, weights_init, weights_value):
        if input_dim < 1 or any(n_units < 1 for n_units in units_per_layer):
            raise ValueError("input_dim and every value in units_per_layer must be positive")
        if len(units_per_layer) != len(acts):
            raise AttributeError(
                f"Mismatching lengths --> len(units_per_layer)={len(units_per_layer)}; len(acts)={len(acts)}")
        if any(act not in act_funcs.keys() for act in acts):
            raise ValueError("Invalid activation function")

    @property
    def input_dim(self):
        return self.__input_dim

    @property
    def units_per_layer(self):
        return self.__units_per_layer

    @property
    def acts(self):
        return self.__acts

    @property
    def layers(self):
        return self.__layers

    @property
    def params(self):
        """
        Returns a dictionary in which the keys are the net's parameters' names,
        and the corresponding values are the current values of those parameters
        """
        return {
            "input_dim": self.input_dim,
            "units_per_layer": self.units_per_layer,
            "acts": self.acts,
        }

    def forward(self, inp=(2, 2, 2), verbose=False):
        """
        Performs a complete forward pass on the whole NN
        :param verbose:
        :param inp: net's input vector
        :return: net's output
        """
        if isinstance(inp, str):
            raise AttributeError("'inp' must be a vector of numbers, got string")
        inp = np.array(inp)
        if len(inp.shape) == 0:
            inp = np.expand_dims(inp, 0)
        pattern_len = inp.shape[1] if len(inp.shape) > 1 else inp.shape[0]
        if pattern_len != self.input_dim:
            raise AttributeError(f"Mismatching lengths --> len(net_inp) = {len(inp)} ; input_dim = {self.input_dim}")

        if verbose:
            print(f"Net's inputs: {inp}")
        # x represents the data through the network (output of a layer, input of the next layer)
        x = inp
        for layer in self.layers:
            x = layer.forward_pass(x)
        if verbose:
            print(f"Net's output: {x}")
        return x

    def compile(self, opt='sgd', loss='squared', metr=('class_acc',), lrn_rate=0.01):
        if opt not in optimizers or loss not in losses:
            raise AttributeError(f"opt must be within {optimizers.keys()} and loss must be in {losses.keys()}")
        self.__opt = optimizers[opt](nn=self, loss=loss, metr=metr, lrn_rate=lrn_rate)

    def fit(self, inp, target, epochs=1, batch_size=1):
        """
        Execute the training of the network
        :param inp: inputs (training set)
        :param target: list of arrays, each array corresponds to a pattern
            and each of its elements is the target for the i-th output unit
        """
        target = np.array(target)
        inp = np.array(inp)
        target_len = target.shape[1] if len(target.shape) > 1 else 1
        if target_len != len(self.layers[-1].units):
            raise AttributeError(
                f"Mismatching shapes --> target: {target.shape} ; output units: {len(self.layers[-1].units)}")
        n_pattern = inp.shape[0] if len(inp.shape) > 1 else 1
        n_target = target.shape[0] if len(target.shape) > 1 else 1
        assert (n_pattern == n_target)
        self.__opt.optimize(train_set=inp, targets=target, epochs=epochs, batch_size=batch_size)

    def propagate_back(self, dErr_dOut):
        # output_layer = self.layers[-1]
        # output_act = output_layer.act
        # dOut_dNet = np.array([output_act.deriv(u.net) for u in output_layer.units])
        # curr_delta = dErr_dOut * dOut_dNet
        curr_delta = dErr_dOut
        for layer in reversed(self.layers):
            curr_delta = layer.backward_pass(curr_delta)

    def get_struct(self):
        structure = copy.deepcopy(self)
        for layer in structure.layers:
            layer.weights = [0.] * len(layer.weights)
            layer.biases = [0.] * len(layer.biases)
        return structure

    def print_net(self):
        """
        Prints the network's architecture and parameters
        """
        print('Neural Network:')
        for layer in self.layers:
            print('  Layer:')
            for unit in layer.units:
                print(f"\tUnit:\n\t  weights: {unit.w}\n\t  bias: {unit.b}\n\t  activation function: {unit.act.name}")
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates a NN with specified topology, inputs and activation functions'
    )
    parser.add_argument(
        '--input_dim',
        action='store',
        type=int,
        help='The input dimension as number, e.g. 3 means input is an array of 3 elements'
    )
    parser.add_argument(
        '--inputs',
        action='store',
        nargs='+',
        type=float,
        help='Array of inputs. Length must be consistent with input dim. e.g. --inputs 2 1.5 1.1'
    )
    parser.add_argument(
        '--units_per_layer',
        action='store',
        nargs='+',
        type=int,
        help='Array as long as the number of layers. Each item is the number of units for the i-th layer e.g. 3 3 2'
    )
    parser.add_argument(
        '--act_funcs',
        action='store',
        nargs='+',
        type=str,
        help=f"List of activation function names, one for each layer. Names to be chosen among {list(act_funcs.keys())}"
    )
    parser.add_argument(
        '--targets',
        action='store',
        nargs='+',
        type=float,
        help="The target outputs"
    )
    parser.add_argument(
        '--weights_init',
        action='store',
        type=str,
        help="The type of weights initialization"
    )
    parser.add_argument(
        '--weights_value',
        action='store',
        type=float,
        help="The value to which the weights are initialized"
    )
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    # All arguments are optional, but either they're all present or they're all None
    excluded_args = ('verbose', 'targets', 'weights_value')
    if (
            not all(vars(args)[arg] is None for arg in vars(args) if arg not in excluded_args) and
            not all(vars(args)[arg] is not None for arg in vars(args) if arg not in excluded_args)
    ):
        parser.error("All arguments are optional, but either they're all present or they're all None")

    # At this point if an arg is not None, all are not None (except targets & verbose). So we check only one argument
    if args.inputs is not None:
        # Check that lengths of arguments lists are consistent
        if args.input_dim != len(args.inputs):
            parser.error("'inputs' vector must have a length equal to 'input_dim'")
        if len(args.units_per_layer) != len(args.act_funcs):
            parser.error("'units_per_layer' vector and 'act_funcs' must have the same length")
        if args.targets is not None:
            if len(args.targets) != args.units_per_layer[-1]:
                parser.error("the length of 'targets' vector must be equal to the number of units in the output layer")

    # Create the net object
    if args.inputs is None:  # one check is enough because either all args are None or they're all not None
        net = Network()
        net.forward(verbose=args.verbose)
    else:
        net = Network(
            input_dim=args.input_dim,
            units_per_layer=args.units_per_layer,
            acts=args.act_funcs,
            weights_init=args.weights_init,
            value=args.weights_value
        )
        if args.targets is None:
            net.forward(args.inputs, verbose=args.verbose)
        else:
            net.compile()
            net.fit(inp=np.array(args.inputs), target=np.array(args.targets))

    if args.verbose:
        net.print_net()
