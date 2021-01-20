import json
import numpy as np
from joblib import Parallel, delayed
import os
from network import Network
from utility import read_cup, get_best_models


class Ensembler:
    def __init__(self, models_filenames: list, retrain: bool, tr_x, tr_y):
        self.models = []
        self.tr_x, self.tr_y, self.test_x = read_cup()

        for filename in models_filenames:
            with open(filename, 'r') as f:
                models_data = json.load(f)

            self.models.append({
                'model': Network(**models_data['model_params']),
                'model_params': models_data['model_params'],
                'train_params': models_data['train_params']
            })

            if not retrain:
                # reshape weights
                for i in range(len(models_data['weights'])):
                    models_data['weights'][i] = np.array(
                        np.reshape(models_data['weights'][i], newshape=np.shape(models_data['weights'][i])))
                    self.models[-1]['model'].weights = models_data['weights']

    def compile(self):
        for model in self.models:
            try:
                model['model'].compile(**model['train_params'])
            except TypeError:
                model['model'].compile(**model['model_params'])

    def fit_serial(self):
        for model in self.models:
            try:
                model['model'].fit(tr_x=self.tr_x, tr_y=self.tr_y, disable_tqdm=False, **model['train_params'])
            except TypeError:
                model['model'].fit(tr_x=self.tr_x, tr_y=self.tr_y, disable_tqdm=False, **model['model_params'])

    def fit_parallel(self):
        try:
            Parallel(n_jobs=os.cpu_count())(delayed(m['model'].fit)(
                tr_x=self.tr_x, tr_y=self.tr_y, disable_tqdm=False, **m['train_params']) for m in self.models)
        except TypeError:
            Parallel(n_jobs=os.cpu_count())(delayed(m['model'].fit)(
                tr_x=self.tr_x, tr_y=self.tr_y, disable_tqdm=False, **m['model_params']) for m in self.models)

    def predict(self):
        res = []
        for i in range(len(self.models)):
            res.append(self.models[i]['model'].predict(inp=self.test_x, disable_tqdm=False))
        return res


if __name__ == '__main__':
    best_models, best_params = get_best_models("cup", coarse=False, n_models=3)
    dir_name = "../ensembler/"
    paths = [dir_name + "model" + str(i) + ".json" for i in range(len(best_models))]
    for i in range(len(best_models)):
        best_models[i].save_model(paths[i])

    ens = Ensembler(models_filenames=paths, retrain=False)
    ens.compile()
    ens.fit_parallel()
    predictions = ens.predict()
