import unittest
from regularizations import regularization
import numpy as np


class TestRegularizations(unittest.TestCase):
    w = np.array([[1, 0.2, -1], [1, 0, 0.5]])
    array_test = np.array([[0.4, 0.08, -0.4], [0.4, 0., 0.2]])

    def test_regularizations(self):
        self.assertAlmostEqual(regularization['l1'].func(self.w, lambd=0.2), 0.7400000000000001)
        self.assertAlmostEqual(regularization['l2'].func(self.w, lambd=0.2), 0.658)
        np.testing.assert_array_almost_equal(regularization['l2'].deriv(self.w, lambd=0.2), self.array_test)


# def test_ValueError(self):
# self.assertRaises(KeyError, regularization['nonexistent_reg'].func, self.w, lambd=0.1)
# self.assertRaises(ValueError, regularization['l1'].func, self.w, lambd=-0.5)


if __name__ == '__main__':
    unittest.main()
