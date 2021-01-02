import unittest
import numpy as np
from functions import act_funcs, losses, metrics, lr_decays


class TestActivationFunctions(unittest.TestCase):
    def test_act_funcs(self):
        np.testing.assert_array_almost_equal(act_funcs['sigmoid'].func([1., 1.]), [0.7310585786, 0.7310585786])
        np.testing.assert_array_almost_equal(act_funcs['relu'].func([2., -3.]), [2., 0.])
        np.testing.assert_array_almost_equal(act_funcs['tanh'].func([1.]), [0.76159415595])
        np.testing.assert_array_equal(act_funcs['leaky_relu'].func([1., -1.]), [1., -0.01])

    def test_act_funcs_derivs(self):
        np.testing.assert_array_almost_equal(act_funcs['sigmoid'].deriv([1.]), [0.196611933241])
        np.testing.assert_array_equal(act_funcs['relu'].deriv([2., -3.]), [1., 0.])
        np.testing.assert_array_almost_equal(act_funcs['tanh'].deriv([1.]), [0.4199743416140])
        np.testing.assert_array_equal(act_funcs['leaky_relu'].deriv([2., -3.]), [1., 0.01])

    # def test_exceptions(self):
    #     # check many combination
    #     # it may be enough to test the function 'check_is_number', but this way we check also that
    #     # we call 'check_is_number' in every activation function and deriv
    #     activation = ['sigmoid', 'relu', 'leaky_relu', 'tanh']
    #     attribute_test = [[1, [1, 2, 3]], 'hello']
    #     for act in activation:
    #         for attr_test in attribute_test:
    #             self.assertRaises(AttributeError, act_funcs[act].func, attr_test)
    #             self.assertRaises(AttributeError, act_funcs[act].deriv, attr_test)


# class TestLossFunctions(unittest.TestCase):
#     predicted = [1, 0, 0, 1]
#     target = [1, 1, 0, 0]
#
#     def test_loss_funcs(self):
#         ground_truth = [0., 0.5, 0., 0.5]
#         for i in range(len(ground_truth)):
#             self.assertEqual(losses['squared'].func(self.predicted, self.target)[i], ground_truth[i])
#
#     def test_loss_funcs_derivs(self):
#         ground_truth = [0., -1., 0., 1.]
#         for i in range(len(ground_truth)):
#             self.assertEqual(losses['squared'].deriv(self.predicted, self.target)[i], ground_truth[i])
#
#     def test_exceptions(self):
#         # test Exception raising with different shapes arrays
#         self.assertRaises(Exception, losses['squared'].func, [0, 0], [0, 0, 0])
#         self.assertRaises(Exception, losses['squared'].deriv, [0, 0], [0, 0, 0])
#
#
# class TestMetrics(unittest.TestCase):
#     def test_metrics(self):
#         self.assertEqual(1, metrics['bin_class_acc'].func(predicted=[1], target=[0.98]))
#         self.assertEqual(0, metrics['bin_class_acc'].func(predicted=[1], target=[0.098]))
#
#
# class TestLRDecays(unittest.TestCase):
#     def test_lr_decays(self):
#         curr_lr = 0.5
#         base_lr = 0.5
#         final_lr = 0.05
#         curr_step = 10
#         limit_step = 100
#         self.assertEqual(
#             lr_decays['linear'].func(curr_lr=curr_lr, base_lr=base_lr, final_lr=final_lr, curr_step=curr_step, limit_step=limit_step),
#             (1 - curr_step / limit_step) * base_lr + curr_step / limit_step * final_lr
#         )
#         curr_step = limit_step + 1
#         self.assertEqual(
#             lr_decays['linear'].func(curr_lr=curr_lr, base_lr=base_lr, final_lr=final_lr, curr_step=curr_step, limit_step=limit_step),
#             final_lr
#         )


if __name__ == '__main__':
    unittest.main()
