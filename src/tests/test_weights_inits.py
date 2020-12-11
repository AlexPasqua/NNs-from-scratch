import unittest
from weights_inits import *


class TestWeightsInits(unittest.TestCase):
    def test_base_func(self):
        self.assertRaises(AttributeError, weights_inits, hello='ciao')

    def test_uniform_init(self):
        self.assertRaises(TypeError, uniform_init, hello=4)

    def test_random_init(self):
        self.assertRaises(TypeError, rand_init, hello='ciao')
    #     self.assertEqual(self.rand_init.type, 'random')
    #     self.assertEqual(len(self.rand_init.w), self.n_weights)
    #     self.assertLessEqual(self.rand_init.b, self.max)
    #     self.assertGreaterEqual(self.rand_init.b, self.min)
    #     for w in self.rand_init.w:
    #         self.assertLessEqual(w, self.max)
    #         self.assertGreaterEqual(w, self.min)


if __name__ == '__main__':
    unittest.main()
