import unittest
from functions import act_funcs


class TestFunctions(unittest.TestCase):
    def test_act_funcs(self):
        self.assertAlmostEqual(act_funcs['sigmoid'].func(1.), 0.7310585786300049)
        self.assertEqual(act_funcs['relu'].func(2.), 2.)
        self.assertEqual(act_funcs['relu'].func(-3.), 0.)
        self.assertAlmostEqual(act_funcs['tanh'].func(1.), 0.7615941559557649)
        self.assertEqual(act_funcs['relu'].func([[[2.]]]), 2.)

    def test_act_funcs_derivs(self):
        self.assertAlmostEqual(act_funcs['sigmoid'].deriv(1.), 0.19661193324148185)
        self.assertEqual(act_funcs['relu'].deriv(2.), 1.)
        self.assertEqual(act_funcs['relu'].deriv(-3.), 0.)
        self.assertAlmostEqual(act_funcs['tanh'].deriv(1.), 0.41997434161402614)

    def test_exceptions(self):
        # check many combination
        # it may be enough to test the function 'check_is_number', but this way we check also that
        # we call 'check_is_number' in every activation function and deriv
        activation = ['sigmoid', 'relu', 'tanh']
        attribute_test = [[1, [1, 2, 3]], 'hello']
        for act in activation:
            for attr_test in attribute_test:
                self.assertRaises(AttributeError, act_funcs[act].func, attr_test)
                self.assertRaises(AttributeError, act_funcs[act].deriv, attr_test)


if __name__ == '__main__':
    unittest.main()
