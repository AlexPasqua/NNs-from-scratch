import unittest
import numpy as np

from metrics import metrics


class TestErrorFunctions(unittest.TestCase):
    predicted = np.array(
        [[1, 0, 0, 1],
         [1, 1, 1, 1]]
    )
    target = np.array(
        [[1, 1, 0, 0],
         [0, 0, 0, 0]]
    )

    def test_functions(self):
        self.assertAlmostEqual(metrics['mee'].func(self.predicted, self.target), 1.22474487139)

    def test_AttributeError(self):
        target_test = np.array(
            [[1, 1, 0, 0],
             [0, 0]]
            , dtype='object')
        self.assertRaises(AttributeError, metrics['mee'].func, self.predicted, target_test)


if __name__ == '__main__':
    unittest.main()
