import json
import numpy as np
from network import Network
from utility import read_monk


class Ensembler:
    models_ensemble_filename = "../ensembler/models_to_ensemble.json"

    def __init__(self, retrain: bool):
        self.models = []
        with open(self.models_ensemble_filename, 'r') as f:
            models_data = json.load(f)


if __name__ == '__main__':
    ens = Ensembler(retrain=True)
    net = Network(17, (4, 1), ('leaky_relu', 'tanh'), 'uniform', limits=(-0.2, 0.2))
    x, y = read_monk("monks-1.train", rescale=True)
    net.compile(opt='sgd', loss='squared', metr='bin_class_acc', lr=0.5, momentum=0.9)
    net.fit(tr_x=x, tr_y=y, epochs=100, batch_size='full', disable_tqdm=False)
    x_test, y_test = read_monk("monks-1.test", rescale=True)
    err, acc = net.evaluate(inp=x_test, targets=y_test, metr='bin_class_acc', loss='squared')
    print(f"Err: {err}\tAcc: {acc}")

    filename = "../ensembler/model0.json"
    net.save_model(filename)
    with open(filename, 'r') as f:
        data = json.load(f)

    for i in range(len(data['weights'])):
        data['weights'][i] = np.array(np.reshape(data['weights'][i], newshape=np.shape(data['weights'][i])))

