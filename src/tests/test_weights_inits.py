import unittest
from weights_inits import inits, UniformInit, Initialization


class TestWeightsInits(unittest.TestCase):
    n_weights = 3
    val_for_unif = 0.1
    min = 0.
    max = 1.
    unif_init = inits['uniform'](val=val_for_unif, n_weights=n_weights)
    rand_init = inits['random'](n_weights=n_weights, lower_lim=min, upper_lim=max)

    def test_uniform_init(self):
        self.assertEqual(self.unif_init.type, 'uniform')
        self.assertEqual(self.unif_init.w, [self.val_for_unif, self.val_for_unif, self.val_for_unif])
        self.assertEqual(self.unif_init.b, self.val_for_unif)

    def test_random_init(self):
        self.assertEqual(self.rand_init.type, 'random')
        self.assertEqual(len(self.rand_init.w), self.n_weights)
        self.assertLessEqual(self.rand_init.b, self.max)
        self.assertGreaterEqual(self.rand_init.b, self.min)
        for w in self.rand_init.w:
            self.assertLessEqual(w, self.max)
            self.assertGreaterEqual(w, self.min)


if __name__ == '__main__':
    unittest.main()
