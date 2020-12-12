import unittest
from weights_inits import *


class TestWeightsInits(unittest.TestCase):
    def test_base_func(self):
        self.assertRaises(AttributeError, weights_inits, hello='ciao')

    def test_uniform_init(self):
        self.assertRaises(TypeError, uniform_init, hello=4)

    def test_random_init(self):
        self.assertRaises(TypeError, rand_init, hello='ciao')


if __name__ == '__main__':
    unittest.main()
