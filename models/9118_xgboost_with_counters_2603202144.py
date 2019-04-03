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

class CountersGenerator(GenericDataPreprocessor):
    def __init__(self):
        self.counters = np.load(os.path.join(
            'data/', 'value_counters.npy'))

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        for i, c in enumerate(X.columns):
            X['count_' + c] = X[c].map(self.counters[i]).astype(np.int32)
        return X

dl_params = {
    'target': "target",
    'id': "ID_code"
}
data_loader = DataLoader(args.data, **dl_params)
data_loader.preprocess(DropColumns, CountersGenerator, ToNumpy)
data_loader.generate_split(StratifiedKFold,
    n_splits=10, shuffle=True, random_state=42)
## << Read and preprocess data

## >> Create and train model
import xgboost as xgb
from sklearn.metrics import roc_auc_score

model_params = {
    'name':          "xgboost_counters_10_folds",
    'fit':           "fit",
    'predict':       "predict_proba",
    'pred_col':      1,
    'online_val':    "eval_set"
}

xgboost_params = {
    'gpu_id': 0,
    'max_bin': 64,
    'tree_method': 'gpu_hist',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'silent': True,
    'booster': 'gbtree',
    'grow_policy': 'lossguide',
    'n_jobs': -1,
    'learning_rate': 0.01,
    'n_estimators': 20000,
    'max_depth': 5,
    'max_delta_step': 5,
    'colsample_bylevel': 0.9,
    'colsample_bytree': 0.95,
    'subsample': 0.8,
    'gamma': 1.5,
    'max_leaves': 10,
    'min_child_weight': 50,
    'reg_alpha': 0.6, # L1 regularization
    'reg_lambda': 50, # L2 regularization
    'seed': 42
}

model = ModelLoader(xgb.XGBClassifier, model_params, **xgboost_params)

fit_params = {
    'early_stopping_rounds': 2500,
    'verbose': 1000
}
predict_params = {}

results = model.run(data_loader, roc_auc_score, fit_params,
    predict_params, verbose=True)

if args.save:
    current_file_path = os.path.abspath(__file__) # to save this .py file
    model.save(data_loader, results, current_file_path, args.preds, args.models)
## << Create and train model
