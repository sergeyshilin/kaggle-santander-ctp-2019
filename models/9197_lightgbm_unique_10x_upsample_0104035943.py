import os
import argparse
import shutil
import numpy as np
import pandas as pd

from data import DataLoader
from data.preprocessors import GenericDataPreprocessor, ToNumpy
from model import ModelLoader, GenericModel

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

class IsUnique(GenericDataPreprocessor):
    def __init__(self):
        self.counters = np.load(os.path.join(
            'data/', 'value_counters.npy'))

    def fit_transform(self, X, y=None):
        return X

    def transform(self, X):
        var_cols = ['var_{}'.format(x) for x in range(200)]

        for i, c in enumerate(var_cols):
            X['count_' + c] = (X[c].map(self.counters[i]) == 1)

        return X

dl_params = {
    'target': "target",
    'id': "ID_code"
}
data_loader = DataLoader(args.data, **dl_params)
data_loader.preprocess(DropColumns, IsUnique)
data_loader.generate_split(StratifiedKFold,
    n_splits=5, shuffle=True, random_state=42)
## << Read and preprocess data

## >> Create and train model
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

model_params = {
    'name':          "lightgbm_unique_10x_upsample",
    'fit':           "fit",
    'predict':       "predict"
}

class LightGbmTrainer(GenericModel):
    def __init__(self):
        self.lgb_params = {
            "device": "gpu",
            "max_bin" : 63,
            "gpu_use_dp" : False,
            "objective" : "binary",
            "metric" : "auc",
            "boosting": 'gbdt',
            "max_depth" : 4,
            "num_leaves" : 13,
            "learning_rate" : 0.01,
            "bagging_freq": 10,
            "bagging_fraction" : 0.8,
            "feature_fraction" : 0.95,
            "min_data_in_leaf": 80,
            "tree_learner": "serial",
            "lambda_l1" : 5,
            "lambda_l2" : 5,
            "bagging_seed" : 42,
            "verbosity" : 0,
            "seed": 42
        }

    def fit(self, train, cv):
        x_tr, y_tr = train
        x_cv, y_cv = cv
        trn_data = lgb.Dataset(x_tr, label=y_tr)
        val_data = lgb.Dataset(x_cv, label=y_cv)
        evals_result = {}
        self.model = lgb.train(self.lgb_params,
                        trn_data,
                        100000,
                        valid_sets = [trn_data, val_data],
                        early_stopping_rounds=3000,
                        verbose_eval = 1000,
                        evals_result=evals_result)

    def predict(self, test):
        return self.model.predict(test)


class UpsamplingPreprocessor(GenericDataPreprocessor):
    def __init__(self):
        self.times = 10

    # Data augmentation
    def augment_class(self, X):
        X_new = X.copy()
        ids = np.arange(X.shape[0])

        for c in range(X.shape[1]):
            np.random.shuffle(ids)
            X_new[:,c] = X[ids][:,c]

        return X_new

    def augment(self, X, y, t=2):
        np.random.seed(42)

        t_pos = t
        t_neg = t // 2

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
        X_augmented, y = self.augment(X.values, y, t=self.times)
        return pd.DataFrame(X_augmented, columns=var_cols), y

    def transform(self, X):
        return X

class IsUniqueOnline(GenericDataPreprocessor):
    def __init__(self):
        self.counters = np.load(os.path.join(
            'data/', 'value_counters.npy'))

    def fit_transform(self, X, y=None):
        return self.transform(X), y

    def transform(self, X):
        var_cols = ['var_{}'.format(x) for x in range(200)]

        for i, c in enumerate(var_cols):
            X['count_' + c] = (X[c].map(self.counters[i]) == 1)

        return X


model = ModelLoader(LightGbmTrainer, model_params)
model.preprocess_online(UpsamplingPreprocessor, IsUniqueOnline)
results = model.run(data_loader, roc_auc_score, {}, {}, verbose=True)

if args.save:
    current_file_path = os.path.abspath(__file__) # to save this .py file
    model.save(data_loader, results, current_file_path, args.preds, args.models)
## << Create and train model
