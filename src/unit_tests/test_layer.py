import unittest
import numpy as np
from network.layer import Layer
from network.unit import Unit


class TestLayer(unittest.TestCase):
    n_units = 3
    value = 0.2
    layer = Layer(fanin=3, n_units=n_units, act='relu', init_type='uniform', value=value)

    def test_creation(self):
        lower_lim = 10
        upper_lim = 12
        layer1 = self.layer
        layer2 = Layer(fanin=3, n_units=self.n_units, act='relu', init_type='random', lower_lim=lower_lim, upper_lim=upper_lim)
        self.assertEqual(self.n_units, len(layer1.units))
        self.assertEqual(self.n_units, len(layer2.units))
        for unit_index in range(len(layer1.units)):
            unit1 = layer1.units[unit_index]
            unit2 = layer2.units[unit_index]
            self.assertEqual(self.value, unit1.b)
            self.assertGreaterEqual(unit2.b, lower_lim)
            self.assertLess(unit2.b, upper_lim)
            for weight_index in range(len(unit1.w)):
                self.assertEqual(self.value, unit1.w[weight_index])
                self.assertGreaterEqual(unit2.w[weight_index], lower_lim)
                self.assertLess(unit2.w[weight_index], upper_lim)

    def test_forward_pass(self):
        inp = [0.5] * self.n_units
        correct_out = []
        for i in range(self.n_units):
            unit = self.layer.units[i]
            correct_out.append(self.layer.act.func(np.dot(unit.w, inp) + unit.b))
            self.assertAlmostEqual(
                correct_out[-1],
                unit.output(inp))
        self.layer.forward_pass(inp)
        for i in range(self.n_units):
            self.assertAlmostEqual(self.layer.outputs[i], correct_out[i])

    def test_backward_pass(self):
        fanin = 2
        layer = Layer(fanin=fanin, n_units=2, act='relu', init_type='uniform', value=0.5)
        layer.forward_pass(inp=[0.5] * fanin)  # needed to initialize outputs and weighted sums
        upstream_delta = np.array([0.5] * len(layer.units))
        res = layer.backward_pass(upstream_delta)
        for i in range(len(layer.units)):
            dOut_dNet = layer.act.deriv(layer.units[i].net)
            delta = dOut_dNet * upstream_delta[i]
            new_upstream_delta = 0.
            for weight in layer.units[i].w:
                val = delta * weight
                self.assertEqual(val, 0.25)
                new_upstream_delta += val
            self.assertEqual(new_upstream_delta, 0.5)

    def test_exceptions(self):
        self.assertRaises(AttributeError, Layer, n_units=2)  # if not all required arguments are passed


if __name__ == '__main__':
    unittest.main()
