import unittest
import numpy as np
from error_functions import err_funcs


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
        self.assertAlmostEqual(err_funcs['mee'].func(self.predicted, self.target), 1.22474487139)

    def test_exceptions(self):
        self.assertRaises(Exception, err_funcs['mee'].func, [0, 0], [0, 0, 0])


if __name__ == '__main__':
    unittest.main()
