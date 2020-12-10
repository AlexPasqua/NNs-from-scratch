import unittest
from regularizations import regularization


class TestRegularizations(unittest.TestCase):
    w = [1, 0.2, -1]

    def test_regularizations(self):
        self.assertAlmostEqual(regularization(self.w, lambd=0.2, reg_type='l1'), 0.44000000000000006)
        self.assertAlmostEqual(regularization(self.w, lambd=0.2, reg_type='l2'), 0.40800000000000003)

    def test_ValueError(self):
        self.assertRaises(ValueError, regularization, self.w, 0.1, 'nonexistent_reg')
        self.assertRaises(ValueError, regularization, self.w, -0.5, 'l1')


if __name__ == '__main__':
    unittest.main()
