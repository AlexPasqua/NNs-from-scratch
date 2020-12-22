import unittest
import numpy as np
from network.unit import Unit


class TestUnit(unittest.TestCase):
    value = 0.2
    n_weights = 3
    unit = Unit(act='relu', init_type='uniform', value=value, n_weights=n_weights)

    def test_creation(self):
        self.assertEqual(self.value, self.unit.b)
        for weight in self.unit.w:
            self.assertEqual(self.value, weight)

    def test_exceptions(self):
        self.assertRaises(TypeError, Unit, act='relu')
        self.assertRaises(TypeError, Unit, init_type='random')


if __name__ == '__main__':
    unittest.main()
