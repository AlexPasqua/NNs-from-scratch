import unittest
from network import Network
from ensembler import Ensembler
from utility import read_cup


class TestEnsembler(unittest.TestCase):
    def test_weight_loading(self):
        net = Network(10, (5, 2), ('leaky_relu', 'identity'), 'uniform', limits=(-0.002, 0.002))
        x, y, _ = read_cup()
        net.compile(opt='sgd', loss='squared', metr='euclidean', lr=0.002, momentum=0.7)
        net.fit(tr_x=x, tr_y=y, epochs=50, batch_size='full', disable_tqdm=False)
        filename = "test_model.json"
        net.save_model(filename)
        ens = Ensembler((filename, filename), retrain=False)

        # test weights loading
        for i in range(len(net.layers)):
            for j in range(len(net.weights[i])):
                for k in range(len(net.weights[i][j])):
                    assert net.weights[i][j][k] == ens.models[0]['model'].weights[i][j][k]

        # test compile
        ens.compile()

        # test fit
        ens.fit_serial()
        # ens.fit_parallel()


if __name__ == '__main__':
    unittest.main()
