import unittest
import numpy as np
from optimizers import optimizers
from network.network import Network


class TestOptimizers(unittest.TestCase):
    gd = optimizers['gd'](
        Network(input_dim=2, units_per_layer=(2, 1), acts=('relu', 'sigmoid')),
        'squared',
        'bin_class_acc',
        lrn_rate=0.2)

    # def test_gd(self):
    #     training_set = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    #     training_labels = np.array([[1], [1], [0], [0]])
    #     self.gd.optimize(
    #         tr_x=training_set,
    #         tr_y=training_labels,
    #         epochs=2,
    #         batch_size=1)


if __name__ == '__main__':
    unittest.main()
