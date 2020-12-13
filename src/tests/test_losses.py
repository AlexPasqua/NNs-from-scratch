import unittest

from src.losses import losses


class TestLosses(unittest.TestCase):
    predicted = [1, 0, 0, 1]
    target = [1, 1, 0, 0]

    def test_functions(self):
        ground_truth = [0., 0.5, 0., 0.5]
        for i in range(len(ground_truth)):
            self.assertEqual(
                losses['squared'].func(self.predicted, self.target)[i],
                ground_truth[i]
            )

    def test_derivatives(self):
        ground_truth = [0., -1., 0., 1.]
        for i in range(len(ground_truth)):
            self.assertEqual(
                losses['squared'].deriv(self.predicted, self.target)[i],
                ground_truth[i]
            )

    def test_exceptions(self):
        # test Exception raising with different shapes arrays
        self.assertRaises(Exception, losses['squared'].func, [0, 0], [0, 0, 0])
        self.assertRaises(Exception, losses['squared'].deriv, [0, 0], [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
