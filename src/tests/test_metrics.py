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
        acc = np.array([0])
        predicted = np.array([[0], [1], [1], [0.8]])
        target = np.array([[1], [1], [1], [1]])
        for i in range(len(predicted)):
            acc += metrics['class_acc'].func(predicted[i], target[i])
        self.assertEqual(3, acc)
        acc = acc / float(len(predicted))
        self.assertEqual(0.75, acc)

    def test_AttributeError(self):
        target_test = np.array(
            [[1, 1, 0, 0],
             [0, 0]]
            , dtype='object')
        self.assertRaises(AttributeError, metrics['mee'].func, self.predicted, target_test)


if __name__ == '__main__':
    unittest.main()
