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
            })

            if not retrain:
                # reshape weights and load them
                for i in range(len(models_data['weights'])):
                    models_data['weights'][i] = np.array(
                        np.reshape(models_data['weights'][i], newshape=np.shape(models_data['weights'][i])))
                    self.models[-1]['model'].weights = models_data['weights']

    def compile(self):
        for model in self.models:
            model['model'].compile(**model['model_params'])

    def fit_serial(self, whole=False):
        final_res = []
        for model in self.models:
            if model['model_params']['epochs'] > 400:
                model['model_params']['epochs'] = 400
        if whole:
            tr_x, tr_y, _ = read_cup(int_ts=False)
            for model in self.models:
                res = model['model'].fit(tr_x=tr_x, tr_y=tr_y, disable_tqdm=False, **model['model_params'])
        else:
            for model in self.models:
                res = model['model'].fit(tr_x=self.tr_x, tr_y=self.tr_y, val_x=self.int_ts_x, val_y=self.int_ts_y,
                                         disable_tqdm=False, **model['model_params'])
                plot_curves(tr_loss=res[0], tr_acc=res[1], val_loss=res[2], val_acc=res[3])
                final_res.append([res[1][-1], res[3][-1]])
                print('\n', res[1][-1], '\t', res[3][-1], '\n')
        return final_res

    def fit_parallel(self):
        for m in self.models:
            if m['model_params']['epochs'] > 400:
                m['model_params']['epochs'] = 400
        res = Parallel(n_jobs=os.cpu_count())(delayed(m['model'].fit)(
            tr_x=self.tr_x, tr_y=self.tr_y, val_x=self.int_ts_x, val_y=self.int_ts_y,
            disable_tqdm=False, **m['model_params']) for m in self.models)
        return res

    def predict(self):
        res = []
        for i in range(len(self.models)):
            res.append(self.models[i]['model'].predict(inp=self.test_x, disable_tqdm=False))
        res = np.mean(res, axis=0)
        return res

    def evaluate(self):
        res = []
        for model in self.models:
            res.append(model['model'].evaluate(inp=self.int_ts_x, targets=self.int_ts_y, metr='euclidean',
                                               loss='squared', disable_tqdm=True))
        res = np.mean(res, axis=0)
        return res


if __name__ == '__main__':
    file_names_n_models = {"alex_local_fine_gs_results_cup.json": 2, "alex_cloud_fine_gs_results_cup.json": 3,
                           "gaetano_2_fine_gs_results_cup.json": 1, "gaetano_cloud_coarse_gs_results_cup.json": 1}
    ens_models = []
    for fn, n_models in file_names_n_models.items():
        best_models, best_params = get_best_models("cup", n_models=n_models, fn=fn)
        for i in range(len(best_models)):
            ens_models.append({'model': best_models[i], 'params': best_params[i]})

    # write models on file
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

    ens.fit_serial(whole=False)

    # writes models with updated weights
    dir_name = "../ensembler/"
    paths = [dir_name + "model" + str(i) + ".json" for i in range(len(ens_models))]
    for i in range(len(ens_models)):
        ens_models[i]['model'].save_model(paths[i])

    evs = ens.evaluate()
    print(f"Loss: {evs[0]}\tMetr: {evs[1]}")

    # retrain on the whole training set
    ens = Ensembler(models_filenames=paths, retrain=False)
    ens.compile()
    final_res = ens.fit_serial(whole=True)
    print("Average development MEE - average internal test MEE:")
    print(np.mean(final_res, axis=0))

    # preds = ens.predict()
    # with open("../cup_predictions.csv", "w") as f:
    #     for i in range(len(preds)):
    #         print(str(i + 1) + ',' + str(preds[i][0]) + ',' + str(preds[i][1]), file=f)
