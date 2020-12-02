import unittest
import numpy as np
from losses import losses


class TestLosses(unittest.TestCase):
    predicted = [1, 0, 0, 1]
    target = [1, 1, 0, 0]

    def test_functions(self):
        self.assertEqual(losses['mse'].func(self.predicted, self.target), 1.5)
        self.assertAlmostEqual(losses['mee'].func(self.predicted, self.target), 1.224744871391589)

    def test_derivatives(self):
        self.assertEqual(losses['mse'].deriv(self.predicted, self.target), 2.)
        self.assertAlmostEqual(losses['mee'].deriv(self.predicted, self.target), 0.8164965809277261)

    def test_exceptions(self):
        # test Exception raising with different shapes arrays
        self.assertRaises(Exception, losses['mse'].func, [0, 0], [0, 0, 0])
        self.assertRaises(Exception, losses['mse'].deriv, [0, 0], [0, 0, 0])
        self.assertRaises(Exception, losses['mee'].func, [0, 0], [0, 0, 0])
        self.assertRaises(Exception, losses['mee'].deriv, [0, 0], [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
