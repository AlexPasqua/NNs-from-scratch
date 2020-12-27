import unittest
from regularizations import regularization
import numpy as np


class TestRegularizations(unittest.TestCase):
    def test_regularizations(self):
        lambd = 0.2
        w = np.array([[1, 0.2, -1], [1, 0, 0.5]])
        l1_deriv = np.array([[1, 1, -1], [1, 0, 1]]) * lambd
        l2_deriv = np.array([[0.4, 0.08, -0.4], [0.4, 0., 0.2]])
        self.assertAlmostEqual(regularization['l1'].func(w, lambd=lambd), 0.740000)
        self.assertAlmostEqual(regularization['l2'].func(w, lambd=lambd), 0.658)
        np.testing.assert_array_almost_equal(regularization['l1'].deriv(w, lambd=lambd), l1_deriv)
        np.testing.assert_array_almost_equal(regularization['l2'].deriv(w, lambd=lambd), l2_deriv)


# def test_ValueError(self):
# self.assertRaises(KeyError, regularization['nonexistent_reg'].func, self.w, lambd=0.1)
# self.assertRaises(ValueError, regularization['l1'].func, self.w, lambd=-0.5)


if __name__ == '__main__':
    unittest.main()
