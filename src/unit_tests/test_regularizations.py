import unittest
from regularizations import regularization


class TestRegularizations(unittest.TestCase):
    w = [1, 0.2, -1]

    def test_regularizations(self):
        self.assertAlmostEqual(regularization['l1'].func(self.w, lambd=0.2), 0.44000)
        self.assertAlmostEqual(regularization['l2'].func(self.w, lambd=0.2), 0.40800)

    def test_ValueError(self):
        # self.assertRaises(KeyError, regularization['nonexistent_reg'].func, self.w, lambd=0.1)
        self.assertRaises(ValueError, regularization['l1'].func, self.w, lambd=-0.5)


if __name__ == '__main__':
    unittest.main()
