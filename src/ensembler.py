import csv
import json
import numpy as np
from joblib import Parallel, delayed
import os
from network import Network
from utility import read_cup, get_best_models, plot_curves
from model_selection import cross_valid


class Ensembler:
    def __init__(self, models_filenames: list, retrain: bool):
        self.models = []
        self.tr_x, self.tr_y, self.int_ts_x, self.int_ts_y, self.test_x = read_cup(int_ts=True)

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
        for m in self.models:
            if m['model_params']['epochs'] > 400:
                m['model_params']['epochs'] = 400

        for model in self.models:
            try:
                model['model'].fit(tr_x=self.tr_x, tr_y=self.tr_y, val_x=self.int_ts_x, val_y=self.int_ts_y,
                                   disable_tqdm=False, **model['train_params'])
            except TypeError:
                model['model'].fit(tr_x=self.tr_x, tr_y=self.tr_y, val_x=self.int_ts_x, val_y=self.int_ts_y,
                                   disable_tqdm=False, **model['model_params'])
            preds = ens.predict()
            return preds

    def fit_parallel(self):
        for m in self.models:
            if m['model_params']['epochs'] > 400:
                m['model_params']['epochs'] = 400
        try:
            res = Parallel(n_jobs=os.cpu_count())(delayed(m['model'].fit)(
                tr_x=self.tr_x, tr_y=self.tr_y, disable_tqdm=False, **m['train_params']) for m in self.models)
        except TypeError:
            res = Parallel(n_jobs=os.cpu_count())(delayed(m['model'].fit)(
                tr_x=self.tr_x, tr_y=self.tr_y, val_x=self.int_ts_x, val_y=self.int_ts_y,
                disable_tqdm=False, **m['model_params']) for m in self.models)
        return res

    def predict(self):
        res = []
        for i in range(len(self.models)):
            res.append(self.models[i]['model'].predict(inp=self.test_x, disable_tqdm=False))
        # res = np.mean(res, axis=0)
        return res


if __name__ == '__main__':
    ens_models = []
    fn = "alex_local_fine_gs_results_cup.json"
    best_models, best_params = get_best_models("cup", coarse=False, n_models=2, fn=fn)
    for i in range(len(best_models)):
        ens_models.append({'model': best_models[i], 'params': best_params[i]})
    fn = "alex_cloud_fine_gs_results_cup.json"
    best_models, best_params = get_best_models("cup", coarse=False, n_models=3, fn=fn)
    for i in range(len(best_models)):
        ens_models.append({'model': best_models[i], 'params': best_params[i]})
    fn = "gaetano_2_fine_gs_results_cup.json"
    best_models, best_params = get_best_models("cup", coarse=False, n_models=1, fn=fn)
    for i in range(len(best_models)):
        ens_models.append({'model': best_models[i], 'params': best_params[i]})
    fn = "gaetano_cloud_coarse_gs_results_cup.json"
    best_models, best_params = get_best_models("cup", coarse=False, n_models=1, fn=fn)
    for i in range(len(best_models)):
        ens_models.append({'model': best_models[i], 'params': best_params[i]})

    # writes models
    dir_name = "../ensembler/"
    paths = [dir_name + "model" + str(i) + ".json" for i in range(len(ens_models))]
    for i in range(len(ens_models)):
        ens_models[i]['model'].save_model(paths[i])

    ens = Ensembler(models_filenames=paths, retrain=False)
    ens.compile()
    # res = ens.fit_parallel()
    # for r in range(len(res)):
    #     plot_curves(tr_loss=res[r][0], tr_acc=res[r][1], val_loss=res[r][2], val_acc=res[r][3],
    #                 path="ens_model" + str(r) + ".png")

    # res = np.mean(res, axis=0)
    # plot_curves(tr_loss=res[0], tr_acc=res[1], val_loss=res[2], val_acc=res[3], path="mean_ens.png")

    # writes models
    preds = ens.fit_serial()
    dir_name = "../ensembler/"
    paths = [dir_name + "model" + str(i) + ".json" for i in range(len(ens_models))]
    for i in range(len(ens_models)):
        ens_models[i]['model'].save_model(paths[i])

    # preds = ens.predict()
    with open("../cup_pridictions.csv", "w") as f:
        for i in range(np.shape(preds)[1]):
            print(str(i) + ',' + str(preds[0][i][0]) + ',' + str(preds[0][i][1]), file=f)
