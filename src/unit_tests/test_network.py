import unittest
from network import Network


class TestNetwork(unittest.TestCase):
    params = {
        'input_dim': 3,
        'units_per_layer': (3, 1),
        'acts': ('sigmoid', 'sigmoid'),
        'init_type': 'fixed',
        'init_value': 0.2,
    }
    net = Network(**params)

    def test_creation(self):
        self.assertEqual(len(self.params['units_per_layer']), len(self.net.layers))
        self.assertRaises(TypeError, Network, input_dim=2)  # if not all required arguments are passed
        self.assertRaises(ValueError, Network, input_dim=-2, units_per_layer=2, acts='relu', init_type='fixed')
        self.assertRaises(ValueError, Network, input_dim=2, units_per_layer=(2, -2), acts='relu', init_type='fixed')
        self.assertRaises(AttributeError, Network, input_dim=2, units_per_layer=(2, 2), acts='relu', init_type='fixed')

    def test_forward(self):
        net = Network(input_dim=3, units_per_layer=[2, 2], acts=['relu', 'relu'], init_type='fixed', init_value=0.5)
        self.assertRaises(ValueError, net.forward, inp='hello')
        self.assertRaises(ValueError, net.forward, inp=2)
        self.assertRaises(ValueError, net.forward, inp=(1, 1))
        self.assertRaises(Exception, net.forward, inp=('a', 'b', 'c'))
        outputs = net.forward(inp=[[1, 1, 1], [1, 1, 1]])
        for out in outputs:
            for value in out:
                self.assertEqual(2.5, value)

    def test_compile(self):
        self.net.compile(opt='sgd')
        self.assertEqual('sgd', self.net.opt.type)
        self.assertRaises(KeyError, self.net.compile, opt='hello', loss='squared')
        self.assertRaises(KeyError, self.net.compile, opt='sgd', loss='hello')
        self.assertRaises(ValueError, self.net.compile, momentum=-1)
        self.assertRaises(ValueError, self.net.compile, momentum=2)

    def test_fit(self):
        net = Network(input_dim=3, units_per_layer=[6, 2], acts=['relu', 'relu'], init_type='fixed', init_value=0.2)
        self.assertRaises(AttributeError, net.fit, tr_x=[1, 1], tr_y=[1, 2, 3])
        self.assertRaises(AttributeError, net.fit, tr_x=[[1, 1], [1, 1]], tr_y=[[1, 2]])
        self.assertRaises(AttributeError, net.fit, tr_x=[[1, 1], [1, 1]], tr_y=[1, 2], val_x=[1, 1], val_y=[1, 2, 3])
        self.assertRaises(ValueError, net.fit, tr_x=[1, 1], tr_y=[1, 2], val_split=12)

    def test_propagate_back(self):
        # call propagate_back without initializing the net first (with a forward)
        self.assertRaises(TypeError,
                          self.net.propagate_back,
                          dErr_dOut=[0.5] * self.net.layers[-1].n_units,
                          grad_net=self.net.get_empty_struct())
        # initialize the net first
        self.net.forward()
        self.net.propagate_back(dErr_dOut=[0.5] * self.net.layers[-1].n_units, grad_net=self.net.get_empty_struct())

    def test_get_empty_gradnet(self):
        gradnet = self.net.get_empty_struct()
        for layer_index in range(len(self.net.layers)):
            layer = self.net.layers[layer_index]
            for unit_index in range(layer.n_units):
                self.assertEqual(0., gradnet[layer_index]['biases'][unit_index])
                for weight_index in range(len(layer.weights[:, unit_index])):
                    self.assertEqual(0., gradnet[layer_index]['weights'][weight_index][unit_index])


if __name__ == '__main__':
    unittest.main()
