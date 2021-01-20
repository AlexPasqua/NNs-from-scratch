import json
import numpy as np
from network import Network
from utility import read_monk


class Ensembler:
    def __init__(self, models_filenames: list, retrain: bool):
        self.models = []
        for filename in models_filenames:
            with open(filename, 'r') as f:
                models_data = json.load(f)

            self.models.append(Network(**models_data['model_params']))

            if not retrain:
                # reshape weights
                for i in range(len(models_data['weights'])):
                    models_data['weights'][i] = np.array(
                        np.reshape(models_data['weights'][i], newshape=np.shape(models_data['weights'][i])))
                    self.models[-1].weights = models_data['weights']
