import math
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import tqdm as tqdm
from functions import losses, metrics


class Optimizer(ABC):
    """
    Abstract class representing a generic optimizer
    (check 'ABC' documentation for more info about abstract classes in Python)

    Attributes:
        net: ('Network' object) Neural Network to which apply the algorithm
        loss: ('DerivableFunction' object) loss function
        metr: ('Function' object) accuracy function
        lrn_rate: (float) learning rate
    """

    @abstractmethod
    def __init__(self, net, loss, metr, lrn_rate, momentum):
        # makes sure lrn_rate is a init_value between 0 and 1
        if lrn_rate <= 0 or lrn_rate > 1:
            raise ValueError('lrn_rate should be a init_value between 0 and 1, Got:{}'.format(lrn_rate))
        self.__net = net
        self.__loss = losses[loss]
        self.__metric = metrics[metr]
        self.__lrn_rate = lrn_rate
        self.momentum = momentum

    @property
    def net(self):
        return self.__net

    @property
    def loss(self):
        return self.__loss

    @property
    def metr(self):
        return self.__metric

    @property
    def lrn_rate(self):
        return self.__lrn_rate


class GradientDescent(Optimizer, ABC):
    """ Gradient Descent """

    def __init__(self, net, loss, metr, lrn_rate=0.01, momentum=0.):
        super(GradientDescent, self).__init__(net, loss, metr, lrn_rate, momentum)
        self.__type = 'gd'

    @property
    def type(self):
        return self.__type

    def optimize(self, tr_x, tr_y, val_x, val_y, epochs, batch_size=1):
        """
        :param tr_x: (numpy ndarray) input training set
        :param tr_y: (numpy ndarray) targets for each input training pattern
        :param val_x: (numpy ndarray) input validation set
        :param val_y: (numpy ndarray) targets for each input validation pattern
        :param epochs: (int) number of training epochs
        :param batch_size: (int) number of patterns per single batch
        :return:
        """
        if len(tr_x.shape) < 2:
            tr_x = tr_x[np.newaxis, :]
        if len(tr_y.shape) < 2:
            tr_y = tr_y[np.newaxis, :]

        tr_error_values = []
        tr_metric_values = []
        val_error_values = []
        val_metric_values = []
        net = self.net
        momentum_net = net.get_empty_struct()

        # cycle through epochs
        for epoch in tqdm.tqdm(range(epochs), desc="Iterating over epochs"):
            epoch_tr_error = np.array([0.] * len(net.layers[-1].units))
            epoch_tr_metric = np.array([0.] * len(net.layers[-1].units))
            epoch_val_error = np.array([0.] * len(net.layers[-1].units))
            epoch_val_metric = np.array([0.] * len(net.layers[-1].units))

            # shuffle the datasets (training & validation) internally
            indexes = list(range(len(tr_x)))
            np.random.shuffle(indexes)
            tr_x = tr_x[indexes]
            tr_y = tr_y[indexes]
            indexes = list(range(len(val_x)))
            np.random.shuffle(indexes)
            val_x = val_x[indexes]
            val_y = val_y[indexes]

            # cycle through batches
            for batch_index in range(math.ceil(len(tr_x) / batch_size)):
                start = batch_index * batch_size
                end = start + batch_size
                train_batch = tr_x[start: end]
                targets_batch = tr_y[start: end]
                grad_net = net.get_empty_struct()

                # cycle through patterns and targets within a batch
                for pattern, target in zip(train_batch, targets_batch):
                    net_outputs = net.forward(inp=pattern)
                    epoch_tr_error = np.add(epoch_tr_error, self.loss.func(predicted=net_outputs, target=target))
                    epoch_tr_metric = np.add(epoch_tr_metric, self.metr.func(predicted=net_outputs, target=target))
                    # equivalent to the following, but it's deprecated
                    # epoch_tr_error[:] += self.loss.func(predicted=net_outputs, target=target)
                    # epoch_tr_metric[:] += self.metr.func(predicted=net_outputs, target=target)
                    dErr_dOut = self.loss.deriv(predicted=net_outputs, target=target)
                    # set the layers' gradients and add them into grad_net
                    # (emulate pass by reference of grad_net using return and reassign)
                    grad_net = net.propagate_back(dErr_dOut, grad_net)

                # weights update
                for layer_index in range(len(net.layers)):
                    grad_net[layer_index]['weights'] /= float(batch_size)
                    grad_net[layer_index]['biases'] /= float(batch_size)
                    delta_w = self.lrn_rate * grad_net[layer_index]['weights']
                    delta_b = self.lrn_rate * grad_net[layer_index]['biases']
                    momentum_net[layer_index]['weights'] *= self.momentum
                    momentum_net[layer_index]['biases'] *= self.momentum
                    momentum_net[layer_index]['weights'] = np.add(momentum_net[layer_index]['weights'], delta_w)
                    momentum_net[layer_index]['biases'] = np.add(momentum_net[layer_index]['biases'], delta_b)
                    net.layers[layer_index].weights = np.add(net.layers[layer_index].weights,
                                                             momentum_net[layer_index]['weights'])
                    net.layers[layer_index].biases = np.add(net.layers[layer_index].biases,
                                                            momentum_net[layer_index]['biases'])

            # validation
            for pattern, target in zip(val_x, val_y):
                net_outputs = net.forward(inp=pattern)
                epoch_val_error = np.add(epoch_val_error, self.loss.func(predicted=net_outputs, target=target))
                epoch_val_metric = np.add(epoch_val_metric, self.metr.func(predicted=net_outputs, target=target))

            epoch_tr_error = np.sum(epoch_tr_error) / float(len(epoch_tr_error))
            epoch_tr_metric = np.sum(epoch_tr_metric) / float(len(epoch_tr_metric))
            epoch_val_error = np.sum(epoch_val_error) / float(len(epoch_val_error))
            epoch_val_metric = np.sum(epoch_val_metric) / float(len(epoch_val_metric))
            tr_error_values.append(epoch_tr_error / float(len(tr_x)))
            tr_metric_values.append(epoch_tr_metric / float(len(tr_x)))
            val_error_values.append(epoch_val_error / float(len(val_x)))
            val_metric_values.append(epoch_val_metric / float(len(val_x)))

        return tr_error_values, tr_metric_values, val_error_values, val_metric_values


optimizers = {
    'gd': GradientDescent
}
