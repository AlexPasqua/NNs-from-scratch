import unittest
from network import Network
from ensembler import Ensembler
from utility import read_monk


class TestEnsembler(unittest.TestCase):
    def test_weight_loading(self):
        net = Network(17, (4, 1), ('leaky_relu', 'tanh'), 'uniform', limits=(-0.2, 0.2))
        x, y = read_monk("monks-1.train", rescale=True)
        net.compile(opt='sgd', loss='squared', metr='bin_class_acc', lr=0.5, momentum=0.9)
        net.fit(tr_x=x, tr_y=y, epochs=100, batch_size='full', disable_tqdm=True)
        filename = "test_model.json"
        net.save_model(filename)
        ens = Ensembler((filename,), retrain=False)
        for i in range(len(net.layers)):
            for j in range(len(net.weights[i])):
                for k in range(len(net.weights[i][j])):
                    assert net.weights[i][j][k] == ens.models[0].weights[i][j][k]


if __name__ == '__main__':
    unittest.main()
