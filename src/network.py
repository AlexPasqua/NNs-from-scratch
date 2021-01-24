import warnings
import json
from tqdm import tqdm
from layer import Layer
from optimizers import *
from functions import losses


class Network:
    """
    Neural network object
    Attributes:
        input_dim: dimension of input layer
        units_per_layer: tuple of integers that indicated the number of units for each layer (input excluded)
        layers: list of net's layers ('Layer' objects)
        opt: Optimizer object
        params: a dictionary with all the main parameters of the network

    The @property methods are used either to have a "getter" method with no possibility to modify the attribute
    from outside the class, or to return indirect attributes (e.g. "weights", that is a concatenation of the weights
    stored in the network's layers)
    """

    def __init__(self, input_dim, units_per_layer, acts, init_type, **kwargs):
        """
        Constructor
        :param input_dim: the input dimension
        :param units_per_layer: list of layers' sizes as number on units
        :param acts: list of activation function names (one for each layer)
        :param init_type: type of weights initializations (either 'fixed' or 'uniform')
        kwargs may contain arguments for the weights initialization
        """
        if not hasattr(units_per_layer, '__iter__'):
            units_per_layer = [units_per_layer]
            acts = [acts]
        self.__check_attributes(self, input_dim=input_dim, units_per_layer=units_per_layer, acts=acts)

        self.__opt = None
        self.__params = {**{'input_dim': input_dim, 'units_per_layer': units_per_layer, 'acts': acts,
                            'init_type': init_type}, **kwargs}
        self.__layers = []
        layer_inp_dim = input_dim
        for i in range(len(units_per_layer)):
            self.__layers.append(Layer(inp_dim=layer_inp_dim, n_units=units_per_layer[i], act=acts[i],
                                       init_type=init_type, **kwargs))
            layer_inp_dim = units_per_layer[i]

    @staticmethod
    def __check_attributes(self, input_dim, units_per_layer, acts):
        """ Checks if the attributes passed to the constructor are consistent """
        if input_dim < 1 or any(n_units < 1 for n_units in units_per_layer):
            raise ValueError("input_dim and every value in units_per_layer must be positive")
        if len(units_per_layer) != len(acts):
            raise AttributeError(
                f"Mismatching lengths --> len(units_per_layer)={len(units_per_layer)}; len(acts)={len(acts)}")

    @property
    def input_dim(self):
        return self.__params['input_dim']

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

    @property
    def weights(self):
        return [layer.weights.tolist() for layer in self.__layers]

    @weights.setter
    def weights(self, value):
        for i in range(len(value)):
            self.__layers[i].weights = value[i]

    def forward(self, inp=(2, 2, 2)):
        """
        Performs a forward pass on the whole Network
        :param inp: net's input vector/matrix
        :return: net's output vector/matrix
        """
        inp = np.array(inp)
        x = inp
        for layer in self.__layers:
            x = layer.forward_pass(x)
        return x

    def compile(self, opt='sgd', loss='squared', metr='bin_class_acc', lr=0.01, lr_decay=None, limit_step=None,
                decay_rate=None, decay_steps=None, staircase=True, momentum=0., reg_type='l2', lambd=0, **kwargs):
        """
        Prepares the network for training by assigning an optimizer to it and setting its parameters
        :param opt: ('Optimizer' object)
        :param loss: (str) the type of loss function
        :param metr: (str) the type of metric to track (accuracy etc)
        :param lr: (float) learning rate value
        :param lr_decay: type of decay for the learning rate
        :param decay_rate: (float) the higher, the stronger the exponential decay
        :param decay_steps: (int) number of steps taken to decay the learning rate exponentially
        :param staircase: (bool) if True the learning rate will decrease in a stair-like fashion (used w/ exponential)
        :param limit_step: number of steps of weights update to perform before stopping decaying the learning rate
        :param momentum: (float) momentum parameter
        :param lambd: (float) regularization parameter
        :param reg_type: (string) regularization type
        """
        if momentum > 1. or momentum < 0.:
            raise ValueError(f"momentum must be a value between 0 and 1. Got: {momentum}")
        self.__params = {**self.__params, **{'loss': loss, 'metr': metr, 'lr': lr, 'lr_decay': lr_decay,
                                             'limit_step': limit_step, 'decay_rate': decay_rate,
                                             'decay_steps': decay_steps, 'staircase': staircase, 'momentum': momentum,
                                             'reg_type': reg_type, 'lambd': lambd}}
        self.__opt = optimizers[opt](net=self, loss=loss, metr=metr, lr=lr, lr_decay=lr_decay, limit_step=limit_step,
                                     decay_rate=decay_rate, decay_steps=decay_steps, staircase=staircase,
                                     momentum=momentum, reg_type=reg_type, lambd=lambd)

    def fit(self, tr_x, tr_y, val_x=None, val_y=None, epochs=1, batch_size=1, val_split=0, **kwargs):
        """
        Execute the training of the network
        :param tr_x: (numpy ndarray) input training set
        :param tr_y: (numpy ndarray) targets for each input training pattern
        :param val_x: (numpy ndarray) input validation set
        :param val_y: (numpy ndarray) targets for each input validation pattern
        :param batch_size: (integer) the size of the batch
        :param epochs: (integer) number of epochs
        :param val_split: percentage of training data to use as validation data (alternative to val_x and val_y)
        """
        # transform sets to numpy array (if they're not already)
        tr_x, tr_y = np.array(tr_x), np.array(tr_y)

        # use validation data
        if val_x is not None and val_y is not None:
            if val_split != 0:
                warnings.warn(f"A validation split was given, but instead val_x and val_y will be used")
            val_x, val_y = np.array(val_x), np.array(val_y)
            n_patterns = val_x.shape[0] if len(val_x.shape) > 1 else 1
            n_targets = val_y.shape[0] if len(val_y.shape) > 1 else 1
            if n_patterns != n_targets:
                raise AttributeError(f"Mismatching shapes {n_patterns} {n_targets}")
        else:
            # use validation split
            if val_split != 0:
                if val_split < 0 or val_split > 1:
                    raise ValueError(f"val_split must be between 0 and 1, got {val_split}")
                indexes = np.random.randint(low=0, high=len(tr_x), size=math.floor(val_split * len(tr_x)))
                val_x = tr_x[indexes]
                val_y = tr_y[indexes]
                tr_x = np.delete(tr_x, indexes, axis=0)
                tr_y = np.delete(tr_y, indexes, axis=0)

        # check that the shape of the target matches the net's architecture
        if batch_size == 'full':
            batch_size = len(tr_x)
        target_len = tr_y.shape[1] if len(tr_y.shape) > 1 else 1
        n_patterns = tr_x.shape[0] if len(tr_x.shape) > 1 else 1
        n_targets = tr_y.shape[0] if len(tr_y.shape) > 1 else 1
        if target_len != self.layers[-1].n_units or n_patterns != n_targets or batch_size > n_patterns:
            raise AttributeError(f"Mismatching shapes")

        self.__params = {**self.__params, 'epochs': epochs, 'batch_size': batch_size}
        return self.__opt.optimize(tr_x=tr_x, tr_y=tr_y, val_x=val_x, val_y=val_y, epochs=epochs,
                                   batch_size=batch_size, **kwargs)

    def predict(self, inp, disable_tqdm=True):
        """
        Computes the outputs for a batch of patterns, useful for testing w/ a blind test set
        :param inp: batch of input patterns
        :return: array of net's outputs
        :param disable_tqdm: (bool) if True disables the progress bar
        """
        inp = np.array(inp)
        inp = inp[np.newaxis, :] if len(inp.shape) < 2 else inp
        predictions = []
        for pattern in tqdm.tqdm(inp, desc="Predicting patterns", disable=disable_tqdm):
            predictions.append(self.forward(inp=pattern))
        return np.array(predictions)

    def evaluate(self, targets, metr, loss, net_outputs=None, inp=None, disable_tqdm=True):
        """
        Performs an evaluation of the network based on the targets and either the pre-computed outputs ('net_outputs')
        or the input data ('inp'), on which the net will first compute the output.
        If both 'predicted' and 'inp' are None, an AttributeError is raised
        :param targets: the targets for the input on which the net is evaluated
        :param metr: the metric to track for the evaluation
        :param loss: the loss to track for the evaluation
        :param net_outputs: the output of the net for a certain input
        :param inp: the input on which the net has to be evaluated
        :return: the loss and the metric
        :param disable_tqdm: (bool) if True disables the progress bar
        """
        if net_outputs is None:
            if inp is None:
                raise AttributeError("Both net_outputs and inp cannot be None")
            net_outputs = self.predict(inp, disable_tqdm=disable_tqdm)
        metr_scores = np.zeros(self.layers[-1].n_units)
        loss_scores = np.zeros(self.layers[-1].n_units)
        for x, y in tqdm.tqdm(zip(net_outputs, targets), total=len(targets), desc="Evaluating model", disable=disable_tqdm):
            metr_scores = np.add(metr_scores, metrics[metr].func(predicted=x, target=y))
            loss_scores = np.add(loss_scores, losses[loss].func(predicted=x, target=y))
        loss_scores = np.sum(loss_scores) / len(loss_scores)
        metr_scores = np.sum(metr_scores) / len(metr_scores)
        loss_scores /= len(net_outputs)
        metr_scores /= len(net_outputs)
        return loss_scores, metr_scores

    def propagate_back(self, dErr_dOut, grad_net):
        """
        Propagates back the error to update each layer's gradient
        :param dErr_dOut: derivatives of the error wrt the outputs
        :param grad_net: a structure with the same topology of the neural network in question, but used to store the
        gradients. It will be updated and returned back to the caller
        :return: the updated grad_net
        """
        curr_delta = dErr_dOut
        for layer_index in reversed(range(len(self.__layers))):
            curr_delta, grad_w, grad_b = self.__layers[layer_index].backward_pass(curr_delta)
            grad_net[layer_index]['weights'] = np.add(grad_net[layer_index]['weights'], grad_w)
            grad_net[layer_index]['biases'] = np.add(grad_net[layer_index]['biases'], grad_b)
        return grad_net

    def get_empty_struct(self):
        """ :return: a zeroed structure with the same topology of the NN to contain all the layers' gradients """
        struct = np.array([{}] * len(self.__layers))
        for layer_index in range(len(self.__layers)):
            struct[layer_index] = {'weights': [], 'biases': []}
            weights_matrix = self.__layers[layer_index].weights
            weights_matrix = weights_matrix[np.newaxis, :] if len(weights_matrix.shape) < 2 else weights_matrix
            struct[layer_index]['weights'] = np.zeros(shape=weights_matrix.shape)
            struct[layer_index]['biases'] = np.zeros(shape=(len(weights_matrix[0, :])))
        return struct

    def print_topology(self):
        """ Prints the network's architecture and parameters """
        print("Model's topology:")
        print("Units per layer: ", self.__params['units_per_layer'])
        print("Activation functions: ", self.__params['acts'])

    def save_model(self, filename: str):
        """ Saves the model to filename """
        data = {'model_params': self.__params, 'weights': self.weights}
        with open(filename, 'w') as f:
            json.dump(data, f, indent='\t')
