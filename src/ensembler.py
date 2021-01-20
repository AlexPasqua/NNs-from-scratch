import json
from network import Network


class Ensembler:
    models_ensemble_filename = "../ensembler/models_to_ensemble.json"

    def __init__(self, retrain: bool):
        self.models = []
        with open(self.models_ensemble_filename, 'r') as f:
            models_data = json.load(f)


if __name__ == '__main__':
    ens = Ensembler(retrain=True)
    net = Network(10, (5, 2), ('relu', 'relu'), 'uniform', limits=(-0.2, 0.2))
    net.save_model("../ensembler/model0.json")
