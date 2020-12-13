import unittest
import numpy as np

from src.error_functions import err_funcs


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
        self.assertEqual(err_funcs['mse'].func(self.predicted, self.target), 3.)

    def test_AttributeError(self):
        target_test = np.array(
            [[1, 1, 0, 0],
             [0, 0]]
            , dtype='object')
        self.assertRaises(AttributeError, err_funcs['mee'].func, self.predicted, target_test)
        self.assertRaises(AttributeError, err_funcs['mse'].func, self.predicted, target_test)



if __name__ == '__main__':
    unittest.main()
