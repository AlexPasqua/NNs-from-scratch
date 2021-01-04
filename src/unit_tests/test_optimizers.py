import unittest
import numpy as np
from optimizers import optimizers
from network.network import Network


class TestOptimizers(unittest.TestCase):
    gd = optimizers['gd'](
        Network(input_dim=2, units_per_layer=(2, 1), acts=('relu', 'sigmoid'), init_type='uniform', init_value=0.2),
        'squared',
        'bin_class_acc',
        lr=0.2,
        lr_decay='linear',
        limit_step=400,
        momentum=0.5,
        reg_type='l2',
        lambd=0.001
    )

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
