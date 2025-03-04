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

    def fit_transform(self, data):
        return data.drop(['ID_code'], axis=1)

    def transform(self, data):
        return data.drop(['ID_code'], axis=1)

class FeatureEngineering(GenericDataPreprocessor):
    def __init__(self):
        pass

    def fit_transform(self, data):
        data['order'] = np.arange(data.shape[0]) / data.shape[0]
        return data

    def transform(self, data):
        return self.fit_transform(data)

dl_params = {
    'target': "target",
    'id': "ID_code"
}
data_loader = DataLoader(args.data, **dl_params)
data_loader.preprocess(DropColumns, FeatureEngineering)
data_loader.preprocess(ToNumpy, StandardScaler)
data_loader.generate_split(StratifiedKFold,
    n_splits=5, shuffle=True, random_state=42)
## << Read and preprocess data

## >> Create and train model
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score

model_params = {
    'name':     "gaussian",
    'fit':      "fit",
    'predict':  "predict_proba",
    'pred_col': 1
}
model = ModelLoader(GaussianNB, model_params)

fit_params = {}; predict_params = {}
results = model.run(data_loader, roc_auc_score, fit_params,
    predict_params, verbose=True)

if args.save:
    current_file_path = os.path.abspath(__file__) # to save this .py file
    model.save(data_loader, results, current_file_path, args.preds, args.models)
## << Create and train model
