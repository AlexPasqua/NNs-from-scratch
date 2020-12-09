import unittest
from src.network import Network


class TestNetwork(unittest.TestCase):
    net = Network(input_dim=3, units_per_layer=[6, 2], acts=['relu', 'relu'])
    net.compile('sgd', 'squared', lrn_rate=0.01)

    def test_forward(self, net=net):
        # make sure Exception is raised when there is a mismatch between input_dim and inp
        self.assertRaises(Exception, net.forward, inp=(1, 1))
        # make sure Exception is raised when there is a mismatch between targets and outputs
        self.assertRaises(Exception, net.fit, inp=(1, 1, 1), target=(1, 0, 1))

    def test_constructor(self):
        activation = ['relu','sigmoid','tanh','threshold']
        # make sure a ValueError is raised when input_dim or units_per_layer are not positive
        for act_i in activation:
            for act_j in activation:
                self.assertRaises(ValueError, Network, input_dim=-2, units_per_layer=[6, 2], acts=[act_i, act_j])
                self.assertRaises(ValueError, Network, input_dim=2, units_per_layer=[-4, 0], acts=[act_i, act_j])

        #TODO: test ValueError if units in one layer have different act functions


if __name__ == '__main__':
    unittest.main()
