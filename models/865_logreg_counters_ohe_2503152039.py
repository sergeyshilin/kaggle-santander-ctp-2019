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
from sklearn.model_selection import StratifiedShuffleSplit
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
        return self.transform(X)

    def transform(self, X):
        for i, c in enumerate(X.columns):
            X['count_' + c] = X[c].map(self.counters[i]).astype(np.int32)
        return X

class Dummies(GenericDataPreprocessor):
    def __init__(self):
        self.sc = StandardScaler()
        pass

    def fit_transform(self, X, y=None):
        dummies = pd.get_dummies(X[X.columns[200:]], sparse=False)
        X_scaled = self.sc.fit_transform(X[X.columns[:200]])
        return np.hstack((X_scaled, dummies))

    def transform(self, X):
        dummies = pd.get_dummies(X[X.columns[200:]], sparse=False)
        X_scaled = self.sc.transform(X[X.columns[:200]])
        return np.hstack((X_scaled, dummies))

dl_params = {
    'target': "target",
    'id': "ID_code"
}
data_loader = DataLoader(args.data, **dl_params)
data_loader.preprocess(DropColumns, CountersGenerator)
import gc
gc.collect()
data_loader.preprocess(Dummies)
gc.collect()
data_loader.generate_split(StratifiedKFold,
    n_splits=5, shuffle=True, random_state=42)
## << Read and preprocess data

## >> Create and train model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

model_params = {
    'name':     "logreg_counters_ohe",
    'fit':      "fit",
    'predict':  "predict_proba",
    'pred_col': 1
}
logreg_params = {
    'C': 0.9,
    'penalty': 'l2',
    'multi_class': 'ovr',
    'solver': 'liblinear',
    'random_state': 42
}
model = ModelLoader(LogisticRegression, model_params, **logreg_params)

fit_params = {}; predict_params = {}
results = model.run(data_loader, roc_auc_score, fit_params,
    predict_params, verbose=True)

if args.save:
    current_file_path = os.path.abspath(__file__) # to save this .py file
    model.save(data_loader, results, current_file_path,
        args.preds, args.models)
## << Create and train model
