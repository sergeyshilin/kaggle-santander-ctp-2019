import os
import argparse
import shutil
import numpy as np
import pandas as pd

from data import DataLoader
from data.preprocessors import GenericDataPreprocessor, ToNumpy
from model import ModelLoader

parser = argparse.ArgumentParser(description=None)
parser.add_argument('--data', required=True,
    help='Path where to read train.csv and test.csv files')
parser.add_argument('--models', required=True,
    help='Path where to save source code')
parser.add_argument('--preds', required=True,
    help='Path where to save model outputs')
parser.add_argument('--nosave', dest='save', action='store_false')
parser.set_defaults(save=True)
args = parser.parse_args()


## >> Read and preprocess data
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from scipy.linalg import norm

class DropColumns(GenericDataPreprocessor):
    def __init__(self):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        return X.drop(['ID_code'], axis=1)

class GroupStats(GenericDataPreprocessor):
    def __init__(self):
        g_1 = ['var_{}'.format(x) for x in range(12)]
        g_2 = ['var_{}'.format(x) for x in range(12, 126)]
        g_3 = ['var_{}'.format(x) for x in range(126, 200)]
        g_1_1 = g_1[:3]
        g_1_2 = g_1[3:]
        g_2_1 = g_2[:96]
        g_2_2 = g_2[96:]

        self.groups = {
            'g_1': g_1,
            'g_2': g_2,
            'g_3': g_3,
            'g_1_1': g_1_1,
            'g_1_2': g_1_2,
            'g_2_1': g_2_1,
            'g_2_2': g_2_2
        }

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        for g, cols in self.groups.items():
            X['{}_mean'.format(g)] = np.mean(X[cols], axis=1)
            X['{}_std'.format(g)] = np.std(X[cols], axis=1)
            X['{}_median'.format(g)] = np.median(X[cols], axis=1)
            X['{}_sum'.format(g)] = np.sum(X[cols], axis=1)
            X['{}_min'.format(g)] = np.min(X[cols], axis=1)
            X['{}_max'.format(g)] = np.max(X[cols], axis=1)
        return X

# class UnderSampleNegativeTarget(GenericDataPreprocessor):
#     def __init__(self):
#         np.random.seed(12345)
#         self.keep_proba = 0.2

#     def fit_transform(self, X, y=None):
#         neg_ids = np.where(y == 0)[0]
#         pos_ids = np.where(y == 1)[0]
#         keep_size = int(X.shape[0] * self.keep_proba)
#         keep_ids = np.random.choice(neg_ids, size=keep_size)
#         new_ids = np.concatenate((keep_ids, pos_ids))
#         return X[new_ids]

#     def transform(self, X):
#         return X

dl_params = {
    'target': "target",
    'id': "ID_code"
}
data_loader = DataLoader(args.data, **dl_params)
data_loader.preprocess(DropColumns, GroupStats, ToNumpy)
data_loader.generate_split(StratifiedKFold,
    n_splits=5, shuffle=True, random_state=42)
## << Read and preprocess data

## >> Create and train model
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_auc_score

model_params = {
    'name':          "catboost_groupstats",
    'fit':           "fit",
    'predict':       "predict_proba",
    'pred_col':      1,
    'online_val':    "eval_set"
}
catboost_params = {
    'loss_function': "Logloss",
    'eval_metric': "AUC",
    'task_type': "GPU",
    'learning_rate': 0.01,
    'iterations': 70000,
    'l2_leaf_reg': 50,
    'random_seed': 42,
    'od_type': "Iter",
    'depth': 5,
    'early_stopping_rounds': 5000,
    'border_count': 64
    }
model = ModelLoader(CatBoostClassifier, model_params, **catboost_params)

fit_params = {
    'use_best_model': True,
    'verbose': 5000,
    'plot': True
    }
predict_params = {}
results = model.run(data_loader, roc_auc_score, fit_params,
    predict_params, verbose=True)

if args.save:
    current_file_path = os.path.abspath(__file__) # to save this .py file
    model.save(data_loader, results, current_file_path, args.preds, args.models)
## << Create and train model
