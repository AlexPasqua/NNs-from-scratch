import unittest
from weights_inits import inits, UniformInit, Initialization


class TestWeightsInits(unittest.TestCase):
    unif_init = inits['uniform'](val=0.1, n_weights=3)

    def test_uniform_init(self):
        self.assertEqual(self.unif_init.type, 'uniform')
        self.assertEqual(self.unif_init.w, [0.1, 0.1, 0.1])
        self.assertEqual(self.unif_init.b, 0.1)


if __name__ == '__main__':
    unittest.main()
