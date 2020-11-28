import unittest
from activation_functions import act_funcs


class TestActivationFunctions(unittest.TestCase):
    def test_functions(self):
        self.assertAlmostEqual(act_funcs['sigmoid'].func(1.), 0.7310585786300049)
        self.assertEqual(act_funcs['relu'].func(2.), 2.)
        self.assertEqual(act_funcs['relu'].func(-3.), 0.)

    def test_derivatives(self):
        self.assertAlmostEqual(act_funcs['sigmoid'].deriv(1.), 0.19661193324148185)
        self.assertEqual(act_funcs['relu'].deriv(2.), 1.)
        self.assertEqual(act_funcs['relu'].deriv(-3.), 0.)


if __name__ == '__main__':
    unittest.main()
