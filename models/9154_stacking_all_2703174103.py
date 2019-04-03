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
from numpy import fft

predictions = [
    '9128_catboost_true_counters_2503133840.csv',
    '898_catboost_with_fft_ifft_2403012714.csv',
    '8532_dense_nn_1303175322.csv',
    '9128_xgboost_counters_10_folds_2603205754.csv',
    '8649_dense_nn_1503181216.csv',
    '9118_xgboost_with_counters_2603202144.csv',
    '9139_catboost_counters_10_folds_2603222240.csv',
    '8885_gaussian_0603221439.csv',
    '9111_lightgbm_with_counters_2603170158.csv',
    '8996_catboost_2_digs_2203182554.csv',
    '8342_catboost_fft_only_2303190343.csv',
    '8993_catboost_groupstats_2203195424.csv',
    '9063_catboost_with_counts_2503021850.csv',
    '9118_lightgbm_with_counters_2603173935.csv',
    '8597_logreg_1303155326.csv',
    '8885_gaussian_1803140138.csv',
    '8995_catboost_3_digs_2203184204.csv',
    '8885_gaussnb_robust_scaling_2003124102.csv',
    '8513_dense_nn_1303172225.csv',
    '865_logreg_counters_ohe_2503152039.csv'
]

class LoadPredictions(GenericDataPreprocessor):
    def __init__(self):
        pass

    def fit_transform(self, X, y=None):
        global predictions
        preds_train = [os.path.join('predictions', 'train', x)
            for x in predictions]

        for i, p in enumerate(preds_train):
            target = pd.read_csv(p).target
            X['preds_{}'.format(i)] = target

        return X

    def transform(self, X):
        global predictions
        preds_test = [os.path.join('predictions', 'test', x)
            for x in predictions]

        for i, p in enumerate(preds_test):
            target = pd.read_csv(p).target
            X['preds_{}'.format(i)] = target

        return X

class DropColumns(GenericDataPreprocessor):
    def __init__(self):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        global predictions
        preds_cols = ['preds_{}'.format(x) for x in range(len(predictions))]
        return X[preds_cols]

dl_params = {
    'target': "target",
    'id': "ID_code"
}
data_loader = DataLoader(args.data, **dl_params)
data_loader.preprocess(LoadPredictions, DropColumns, ToNumpy)
data_loader.generate_split(StratifiedKFold,
    n_splits=10, shuffle=True, random_state=42)
## << Read and preprocess data

## >> Create and train model
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_auc_score

model_params = {
    'name':          "stacking_all",
    'fit':           "fit",
    'predict':       "predict_proba",
    'pred_col':      1,
    'online_val':    "eval_set"
}
catboost_params = {
    'loss_function': "Logloss",
    'eval_metric': "AUC",
    'task_type': "CPU",
    'learning_rate': 0.01,
    'iterations': 25000,
    'l2_leaf_reg': 100,
    'random_seed': 42,
    'subsample': 0.9,
    'rsm': 0.8, # % of features
    'bootstrap_type': "Bernoulli",
    'od_type': "Iter",
    'depth': 2,
    'early_stopping_rounds': 100,
    'border_count': 254
    }
model = ModelLoader(CatBoostClassifier, model_params, **catboost_params)

fit_params = {
    'use_best_model': True,
    'verbose': 1000,
    'plot': True
    }
predict_params = {}
results = model.run(data_loader, roc_auc_score, fit_params,
    predict_params, verbose=True)

if args.save:
    current_file_path = os.path.abspath(__file__) # to save this .py file
    model.save(data_loader, results, current_file_path, args.preds, args.models)
## << Create and train model
