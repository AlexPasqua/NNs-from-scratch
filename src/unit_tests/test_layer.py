import unittest
import numpy as np
from network.layer import Layer
import datetime


class TestLayer(unittest.TestCase):
    n_units = 5
    value = 0.2
    fanin = 3
    layer = Layer(inp_dim=fanin, n_units=n_units, act='relu', init_type='fixed', lower_lim=-1, upper_lim=1,
                  init_value=value)

    def test_creation(self):
        lower_lim = -2
        upper_lim = 2.2
        layer2 = Layer(
            inp_dim=self.fanin,
            n_units=self.n_units,
            act='relu',
            init_type='uniform',
            lower_lim=lower_lim,
            upper_lim=upper_lim)
        for i in range(len(self.layer.weights)):
            for j in range(len(self.layer.weights[0])):
                self.assertEqual(self.layer.weights[i][j], self.value)
                self.assertGreaterEqual(layer2.weights[i][j], lower_lim)
                self.assertLessEqual(layer2.weights[i][j], upper_lim)

    def test_forward_pass(self):
        inp = [0.5] * 3
        correct_out = self.layer.act.func(np.add(np.matmul(inp, self.layer.weights), self.layer.biases))
        np.testing.assert_array_equal(self.layer.forward_pass(inp), correct_out)

    def test_backward_pass(self):
        fanin = 2
        layer = Layer(inp_dim=fanin, n_units=2, act='relu', init_type='fixed', init_value=0.5)
        layer.forward_pass(inp=[0.5] * fanin)  # needed to initialize outputs and weighted sums
        upstream_delta = np.array([0.5] * layer.n_units)
        res, grdw, grdb = layer.backward_pass(upstream_delta)
        for i in range(layer.n_units):
            dOut_dNet = layer.act.deriv(layer.nets)[i]
            delta = dOut_dNet * upstream_delta[i]
            new_upstream_delta = 0.
            for weight in layer.weights[i]:
                val = delta * weight
                self.assertEqual(val, 0.25)
                new_upstream_delta += val
            self.assertEqual(new_upstream_delta, res[i])
            self.assertEqual(-delta, grdb[i])
            for j in range(layer.inp_dim):
                self.assertEqual(grdw[j][i], -delta * layer.inputs[j])

    def test_exceptions(self):
        self.assertRaises(TypeError, Layer, n_units=2)  # if not all required arguments are passed


if __name__ == '__main__':
    unittest.main()
