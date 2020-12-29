import math
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np
import tqdm as tqdm
from functions import losses, metrics, regs


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
    def __init__(self, net, loss, metr, lrn_rate, momentum, lambd, reg_type):
        # makes sure lrn_rate is a value between 0 and 1
        if lrn_rate <= 0 or lrn_rate > 1:
            raise ValueError('lrn_rate should be a value between 0 and 1, Got:{}'.format(lrn_rate))
        self.__net = net
        self.__loss = losses[loss]
        self.__metric = metrics[metr]
        self.__lrn_rate = lrn_rate
        self.momentum = momentum
        self.lambd = lambd
        self.reg_type = reg_type

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

    def __init__(self, net, loss, metr, lrn_rate=0.01, momentum=0., lambd=0., reg_type='l2'):
        super(GradientDescent, self).__init__(net, loss, metr, lrn_rate, momentum, lambd, reg_type)
        self.__type = 'gd'

    @property
    def type(self):
        return self.__type

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
        metric_values = []
        net = self.net
        momentum_net = net.get_empty_struct()

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
                train_batch = train_set[start: end]
                targets_batch = targets[start: end]
                grad_net = net.get_empty_struct()

                # cycle through patterns and targets within a batch
                for pattern, target in zip(train_batch, targets_batch):
                    net_outputs = net.forward(inp=pattern)

                    # computes penalty term
                    for index in range(len(net.layers) - 1):
                        w_tot = np.concatenate((net.layers[index].weights, net.layers[index + 1].weights))
                    regularization = regs[self.reg_type].func(w=w_tot, lambd=self.lambd)

                    epoch_error[:] += self.loss.func(predicted=net_outputs, target=target) + regularization
                    epoch_metric[:] += self.metr.func(predicted=net_outputs, target=target) + regularization
                    dErr_dOut = self.loss.deriv(predicted=net_outputs, target=target)
                    # set the layers' gradients and add them into grad_net
                    # (emulate pass by reference of grad_net using return and reassign)
                    grad_net = net.propagate_back(dErr_dOut, grad_net)

                    # add up layers' gradients
                    # for i in range(len(net.layers)):
                    #     grad_net.layers[i].weights += net.layers[i].__gradient_w
                    #     grad_net.layers[i].biases += net.layers[i].__gradient_b

                # weights update
                for layer_index in range(len(net.layers)):
                    grad_net[layer_index]['weights'] /= float(batch_size)
                    grad_net[layer_index]['biases'] /= float(batch_size)
                    delta_w = self.lrn_rate * grad_net[layer_index]['weights']
                    delta_b = self.lrn_rate * grad_net[layer_index]['biases']
                    momentum_net[layer_index]['weights'] *= self.momentum
                    momentum_net[layer_index]['biases'] *= self.momentum
                    momentum_net[layer_index]['weights'] += delta_w
                    momentum_net[layer_index]['biases'] += delta_b
                    net.layers[layer_index].weights += momentum_net[layer_index]['weights'] - regs[self.reg_type].deriv(
                                                                                    w=net.layers[layer_index].weights,
                                                                                    lambd=self.lambd)

                    net.layers[layer_index].biases += momentum_net[layer_index]['biases']

            epoch_error = np.sum(epoch_error) / float(len(epoch_error))
            epoch_metric = np.sum(epoch_metric) / float(len(epoch_metric))
            errors.append(epoch_error / float(len(train_set)))
            metric_values.append(epoch_metric / float(len(train_set)))

        # plot learning curve
        plt.plot(range(epochs), errors)
        plt.xlabel('Epochs', fontweight='bold')
        plt.ylabel('loss', fontweight='bold')
        plt.title(f"Eta:{self.lrn_rate}  Alpha:{self.momentum}  Lambda:  Layers:{len(net.units_per_layer)}",
                  fontweight='bold')
        plt.show()

        plt.plot(range(epochs), metric_values)
        plt.xlabel('Epochs', fontweight='bold')
        plt.ylabel('accuracy', fontweight='bold')
        plt.title(f"Eta:{self.lrn_rate}  Alpha:{self.momentum}  Lambda:/empty/  Layers:{len(net.units_per_layer)}",
                  fontweight='bold')
        plt.show()


optimizers = {
    'gd': GradientDescent
}
