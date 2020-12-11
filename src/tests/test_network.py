import unittest
from network import Network
from activation_functions import act_funcs
import random


class TestNetwork(unittest.TestCase):

    def test_constructor(self):
        self.assertRaises(ValueError, Network, input_dim=-2, units_per_layer=[6, 2], acts=['relu', 'relu'])
        self.assertRaises(ValueError, Network, input_dim=2, units_per_layer=[-4, 1], acts=['relu', 'relu'])
        self.assertRaises(ValueError, Network, input_dim=3, units_per_layer=[3, 0], acts=['relu', 'relu'])
        self.assertRaises(ValueError, Network, input_dim=2, units_per_layer=[6, 2], acts=['relu', 'hello'])
        self.assertRaises(AttributeError, Network, input_dim=3, units_per_layer=[3, 2], acts=['relu', 'relu', 'relu'])

    def test_forward(self):
        net = Network(input_dim=3, units_per_layer=[6, 2], acts=['relu', 'relu'])
        self.assertRaises(AttributeError, net.forward, inp='hello')
        self.assertRaises(AttributeError, net.forward, inp=2)
        self.assertRaises(AttributeError, net.forward, inp=(1, 1))
        self.assertRaises(Exception, net.forward, inp=('a', 'b', 'c'))

    def test_compile(self):
        pass

    def test_fit(self):
        net = Network(input_dim=3, units_per_layer=[6, 2], acts=['relu', 'relu'])
        net.compile('sgd', 'squared', lrn_rate=0.01)
        # make sure Exception is raised when there is a mismatch between targets and outputs
        self.assertRaises(Exception, net.fit, inp=(1, 1, 1), target=(1, 0, 1))
        self.assertRaises(AttributeError, net.fit, [1, 1, 1], 'ciao')


if __name__ == '__main__':
    unittest.main()
