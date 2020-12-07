import unittest
from activation_functions import act_funcs, check_is_number


class TestActivationFunctions(unittest.TestCase):
    def test_functions(self):
        self.assertAlmostEqual(act_funcs['sigmoid'].func(1.), 0.7310585786300049)
        self.assertEqual(act_funcs['relu'].func(2.), 2.)
        self.assertEqual(act_funcs['relu'].func(-3.), 0.)
        self.assertAlmostEqual(act_funcs['tanh'].func(1.), 0.7615941559557649)

    def test_derivatives(self):
        self.assertAlmostEqual(act_funcs['sigmoid'].deriv(1.), 0.19661193324148185)
        self.assertEqual(act_funcs['relu'].deriv(2.), 1.)
        self.assertEqual(act_funcs['relu'].deriv(-3.), 0.)
        self.assertAlmostEqual(act_funcs['tanh'].deriv(1.), 0.41997434161402614)

    def test_exceptions(self):
        # check many combination
        # it may be enough to test the function 'check_is_number', but this way we check also that
        # we call 'check_is_number' in every activation function and deriv
        self.assertRaises(AttributeError, act_funcs['sigmoid'].func, 'hello')
        self.assertRaises(AttributeError, act_funcs['sigmoid'].func, [1, 2, 3])
        self.assertRaises(AttributeError, act_funcs['sigmoid'].deriv, 'hello')
        self.assertRaises(AttributeError, act_funcs['sigmoid'].deriv, [1, 2, 3])
        self.assertRaises(AttributeError, act_funcs['relu'].func, 'hello')
        self.assertRaises(AttributeError, act_funcs['relu'].func, [1, 2, 3])
        self.assertRaises(AttributeError, act_funcs['relu'].deriv, 'hello')
        self.assertRaises(AttributeError, act_funcs['relu'].deriv, [1, 2, 3])
        self.assertRaises(AttributeError, act_funcs['tanh'].func, 'hello')
        self.assertRaises(AttributeError, act_funcs['tanh'].func, [1, 2, 3])
        self.assertRaises(AttributeError, act_funcs['tanh'].deriv, 'hello')
        self.assertRaises(AttributeError, act_funcs['tanh'].deriv, [1, 2, 3])


if __name__ == '__main__':
    unittest.main()
