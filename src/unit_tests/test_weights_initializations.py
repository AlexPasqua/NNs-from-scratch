import unittest
from weights_initializations import weights_inits


class TestWeightsInitializations(unittest.TestCase):
    def test_exceptions(self):
        self.assertRaises(AttributeError, weights_inits, foo='foo')
        self.assertRaises(AttributeError, weights_inits)
        self.assertRaises(TypeError, weights_inits, 'positional_argument')  # it takes 0 positional arguments
        self.assertRaises(TypeError, weights_inits, type='uniform')     # missing arguments for uniform initialization
        self.assertRaises(TypeError, weights_inits, type='random')  # missing arguments for random initialization
        self.assertRaises(ValueError, weights_inits, type='random', n_weights=2, upper_lim=-0.2, lower_lim=0.)

    def test_values(self):
        value = 0.2
        self.assertEqual(weights_inits(type='uniform', value=value, n_weights=1), value)
        self.assertLessEqual(weights_inits(type='random', lower_lim=0., upper_lim=value, n_weights=1), value)


if __name__ == '__main__':
    unittest.main()
