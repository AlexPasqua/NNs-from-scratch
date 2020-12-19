""" In this scripts are defined the optimizers used in the project """
import math
import random
from abc import ABC, abstractmethod
import tqdm as tqdm
from losses import losses
from network import *
import numpy as np
import matplotlib.pyplot as plt
from metrics import metrics
from network import *


class Optimizer(ABC):
    """
    Abstract class representing a generic optimizer
    (check 'ABC' documentation for more info about abstract classes in Python)

    Attributes:
        nn: Neural Network --> 'Network' object
        loss: loss function --> 'Function' object
    """

    @abstractmethod
    def __init__(self, loss, metr, lrn_rate=0.01):
        self.__loss = losses[loss]
        self.__metr = metrics[metr]
        self.__lrn_rate = lrn_rate

    @property
    def loss(self):
        return self.__loss

    @property
    def metr(self):
        return self.__metr

    @property
    def lrn_rate(self):
        return self.__lrn_rate


class SGD(Optimizer, ABC):
    """ Stochastic Gradient Descent """

    def __init__(self, nn, loss, metr, lrn_rate=0.01):
        self.__nn = nn
        super(SGD, self).__init__(loss, metr, lrn_rate)
        # makes sure lrn_rate is a value between 0 and 1
        if lrn_rate <= 0 or lrn_rate > 1:
            raise ValueError('lrn_rate should be a value between 0 and 1, Got:{}'.format(lrn_rate))

    def optimize(self, train_set, targets, epochs, batch_size=1):
        """
        :param train_set: (numpy ndarray) network's inputs
        :param targets: (numpy ndarray)
        :param epochs: (int) number of training epochs
        :param batch_size: (int) number of patterns per single batch
        :return:
        """
        if len(train_set.shape) < 2:
            train_set = train_set[np.newaxis, :]
        if len(targets.shape) < 2:
            targets = targets[np.newaxis, :]

        errors = []
        metrics = []
        net = self.__nn

        # cycle through epochs
        for epoch in tqdm.tqdm(range(epochs), desc="Iterating over epochs"):
            # shuffle the dataset
            indexes = list(range(len(targets)))
            np.random.shuffle(indexes)
            train_set = train_set[indexes]
            targets = targets[indexes]

            epoch_error = np.array([0.] * len(net.layers[-1].units))
            epoch_metric = np.array([0.] * len(net.layers[-1].units))

            # cycle through batches
            for batch_index in range(math.ceil(len(train_set) / batch_size)):
                start = batch_index * batch_size
                end = start + batch_size
                train_batch = train_set[start : end]
                targets_batch = targets[start : end]
                grad_net = net.get_struct()

                # cycle through patterns and targets within a batch
                for pattern, target in zip(train_batch, targets_batch):
                    net_outputs = net.forward(inp=pattern)
                    epoch_error[:] += self.loss.func(predicted=net_outputs, target=target)
                    epoch_metric[:] += self.metr.func(predicted=net_outputs, target=target)
                    dErr_dOut = self.loss.deriv(predicted=net_outputs, target=target)
                    net.propagate_back(dErr_dOut)   # set the layers' gradients

                    # add up layers' gradients
                    for i in range(len(net.layers)):
                        grad_net.layers[i].weights += net.layers[i].gradient_w
                        grad_net.layers[i].biases += net.layers[i].gradient_b

                # weights update
                for i in range(len(net.layers)):
                    grad_net.layers[i].weights = np.array(grad_net.layers[i].weights)
                    grad_net.layers[i].biases = np.array(grad_net.layers[i].biases)
                    grad_net.layers[i].weights /= float(batch_size)
                    grad_net.layers[i].biases /= float(batch_size)
                    net.layers[i].weights += self.lrn_rate * grad_net.layers[i].weights
                    net.layers[i].biases += self.lrn_rate * grad_net.layers[i].biases

            epoch_error = np.sum(epoch_error) / float(len(epoch_error))
            epoch_metric = np.sum(epoch_metric) / float(len(epoch_metric))
            errors.append(epoch_error / float(len(train_set)))
            metrics.append(epoch_metric / float(len(train_set)))

        plt.plot(range(epochs), metrics)
        plt.show()

        # OLD BACKPROP --> keep it as reference
        # losses = []
        # for pattern, target in zip(train_set, targets):
        #     output_layer = self.__nn.layers[-1]
        #     output_act = output_layer.act
        #     net_outputs = self.__nn.forward(inp=pattern)
        #
        #     err = self.loss.func(predicted=net_outputs, target=target)
        #     losses.append(err / len(err))
        #     print('Loss: ', np.sum(err) / len(err))
        #
        #     dErr_dOut = self.loss.deriv(predicted=net_outputs, target=target)
        #     dOut_dNet = [output_act.deriv(u.net) for u in output_layer.units]
        #     delta = -dErr_dOut * dOut_dNet
        #     delta_next = delta
        #
        #     # retrieve the inputs of the output layer to compute the weights update for the output layer
        #     out_layer_inputs = self.__nn.layers[-2].outputs if len(self.__nn.layers) > 1 else pattern
        #     dErr_dBiases = -delta
        #     dErr_dWeights = [
        #         -delta[j] * out_layer_inputs[i]
        #         for j in range(len(delta))
        #         for i in range(len(out_layer_inputs))
        #     ]
        #
        #     # variables used for weights and biases updates
        #     # delta_weights: list of lists --> layers x weights_in_layer
        #     # delta_biases: list of lists --> layers x biases_in_layer
        #     # delta_weights = delta_biases = [[]] * len(self.__nn.layers)
        #     delta_weights = [[]] * len(self.__nn.layers)
        #     delta_biases = [[]] * len(self.__nn.layers)
        #     delta_weights[-1] = [-dErr_dWeights[i] for i in range(len(dErr_dWeights))]
        #     delta_biases[-1] = [-dErr_dBiases[j] for j in range(len(dErr_dBiases))]
        #
        #     # scan all layers from the penultimate to the first
        #     for layer_index in reversed(range(len(self.__nn.layers) - 1)):
        #         curr_layer = self.__nn.layers[layer_index]
        #         next_layer = self.__nn.layers[layer_index + 1]
        #         n_curr_units = len(curr_layer.units)  # number of units in the current layer
        #         n_next_units = len(next_layer.units)  # number of units in the next layer
        #         curr_act = curr_layer.act
        #
        #         dOut_dNet = [curr_act.deriv(u.net) for u in curr_layer.units]
        #
        #         delta = [np.dot(delta_next, [u.w[j] for u in next_layer.units]) for j in range(n_curr_units)]
        #         delta = np.multiply(delta, dOut_dNet)
        #         delta_next = delta
        #         # equivalent to:
        #         # delta = np.zeros([len(curr_layer.units)])
        #         # for j in range(len(curr_layer.units)):
        #         #     for l in range(len(next_layer.units)):
        #         #         delta[j] += next_layer.units[l].w[j] * delta_next[l]
        #         #     delta[j] *= dOut_dNet[j]
        #
        #         curr_layer_inputs = self.__nn.layers[layer_index - 1].outputs if layer_index > 0 else pattern
        #         dErr_dBiases = -delta
        #         dErr_dWeights = [
        #             -delta[j] * curr_layer_inputs[i]
        #             for j in range(len(delta))
        #             for i in range(len(curr_layer_inputs))
        #         ]
        #         delta_weights[layer_index] = [-dErr_dWeights[i] for i in range(len(dErr_dWeights))]
        #         delta_biases[layer_index] = [-dErr_dBiases[j] for j in range(len(dErr_dBiases))]
        #
        #     # update weights and biases
        #     for layer_index in range(len(self.__nn.layers)):
        #         curr_layer = self.__nn.layers[layer_index]
        #         curr_layer.weights += self.lrn_rate * np.array(delta_weights[layer_index])
        #         curr_layer.biases += self.lrn_rate * np.array(delta_biases[layer_index])
        #
        # plt.plot(losses)
        # plt.show()


optimizers = {
    'sgd': SGD
}

if __name__ == '__main__':
    opt = optimizers['sgd'](
        Network(
            input_dim=3,
            units_per_layer=[3, 3, 3, 1],
            acts=['sigmoid', 'sigmoid', 'sigmoid', 'sigmoid'],
            weights_init='uniform',
            weights_value=0.5
        ),
        loss='squared',
        lrn_rate=1
    )
    n_patterns = 20
    inputs = np.reshape([0.1, 0.1, 0.1] * n_patterns, newshape=(n_patterns, 3))
    targets = np.reshape([1] * n_patterns, newshape=(n_patterns, 1))
    opt.optimize(train_set=np.array(inputs),
                 targets=np.array(targets),
                 epochs=3,
                 batch_size=1)
