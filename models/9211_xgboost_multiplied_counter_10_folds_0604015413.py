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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

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
        return X

    def transform(self, X):
        var_cols = ['var_{}'.format(x) for x in range(200)]

        for i, c in enumerate(var_cols):
            X['varcount_' + c] = X[c].map(self.counters[i]).astype(np.int32)
            X['unique_' + c] = (X['varcount_' + c] == 1)
            X['varcount_' + c] = X['varcount_' + c] * X[c]

        return X

dl_params = {
    'target': "target",
    'id': "ID_code"
}
data_loader = DataLoader(args.data, **dl_params)
data_loader.preprocess(DropColumns, CountersGenerator)
data_loader.generate_split(StratifiedKFold,
    n_splits=10, shuffle=True, random_state=42)
## << Read and preprocess data

## >> Create and train model
import xgboost as xgb
from sklearn.metrics import roc_auc_score

model_params = {
    'name':          "xgboost_multiplied_counter_10_folds",
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
    'learning_rate': 0.005,
    'n_estimators': 150000,
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

class UpsamplingPreprocessor(GenericDataPreprocessor):
    def __init__(self):
        self.times = 10
        self.neg_class_balancer = 3
        self.random_seed = 42

    # Data augmentation
    def augment_class(self, X):
        X_new = X.copy()

        for c in range(X.shape[1]):
            np.random.shuffle(X_new[:, c])

        return X_new

    def augment(self, X, y):
        np.random.seed(self.random_seed)

        t_pos = self.times
        t_neg = self.times // self.neg_class_balancer

        X_pos_orig = X[y == 1]
        X_neg_orig = X[y == 0]
        X_pos = np.zeros((t_pos, *X_pos_orig.shape), dtype=X.dtype)
        X_neg = np.zeros((t_neg, *X_neg_orig.shape), dtype=X.dtype)

        for i in range(t_pos):
            X_pos[i] = self.augment_class(X_pos_orig)

        for i in range(t_neg):
            X_neg[i] = self.augment_class(X_neg_orig)

        X_pos = np.vstack(X_pos)
        X_neg = np.vstack(X_neg)
        y_pos = np.ones(X_pos.shape[0])
        y_neg = np.zeros(X_neg.shape[0])
        X = np.vstack((X, X_pos, X_neg))
        y = np.concatenate((y, y_pos, y_neg))

        return X, y

    def fit_transform(self, X, y=None):
        var_cols = ['var_{}'.format(x) for x in range(200)]
        X_augmented, y = self.augment(X.values, y)
        return pd.DataFrame(X_augmented, columns=var_cols), y

    def transform(self, X):
        return X

class CountersGeneratorOnline(GenericDataPreprocessor):
    def __init__(self):
        self.counters = np.load(os.path.join(
            'data/', 'value_counters.npy'))

    def fit_transform(self, X, y=None):
        return self.transform(X), y

    def transform(self, X):
        var_cols = ['var_{}'.format(x) for x in range(200)]

        for i, c in enumerate(var_cols):
            X['varcount_' + c] = X[c].map(self.counters[i]).astype(np.int32)
            X['unique_' + c] = (X['varcount_' + c] == 1)
            X['varcount_' + c] = X['varcount_' + c] * X[c]

        return X

model.preprocess_online(UpsamplingPreprocessor, CountersGeneratorOnline)


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
