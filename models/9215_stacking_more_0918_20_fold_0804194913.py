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

my_best = [
    '9211_xgboost_multiplied_counter_10_folds_0604015413.csv',
    '9202_separate_feature_models_threshold_0.84_10fold_04042353.csv',
    '9223_catboost_unique_10_fold_20x_upsample_0104001611.csv',
    '9208_separate_feature_models_threshold_0.80_10fold_07041509.csv',
    '9222_catboost_multiplied_counter_10_folds_0504034846.csv',
    '9215_catboost_unique_10x_upsample_3_balancer_0204145755.csv',
    '9211_catboost_unique_with_10x_upsample_3103172847.csv',
    '9199_xgboost_unique_10x_upsample_0104062006.csv',
    '9222_catboost_poisson_bootstrap_10_folds_0804042543.csv',
    '9218_catboost_excluded_features_10_folds_0804184027.csv',
    '9211_catboost_unbalanced_weights_0304111611.csv',
    '9219_catboost_best_parameters_10_folds_0704212230.csv',
    '9212_catboost_unique_10x_upsample_5_balancer_0204172825.csv',
    '9197_lightgbm_unique_10x_upsample_0104035943.csv',
    '9210_lightgbm_multiplied_counter_10_folds_0604220718.csv',
    '9218_catboost_multiplied_counter_0304011056.csv'
]

new_best = [
    'nn_0_0.92345_oof.npy',
    'nn_aug_0.92303_12_oof.npy',
    'nn_3_0.92114_oof.npy',
    'nn_3_0.92235_oof.npy',
    'nn_4_0.92292_oof.npy',
    'nn_0_0.92269_oof.npy',
    'nn_3_0.92287_oof.npy',
    'nn_5_0.92258_oof.npy',
    'nn_1_0.92294_oof.npy',
    'nn_2_0.92128_oof.npy',
    'nn_1_0.92273_oof.npy',
    'nn_0_0.92226_oof.npy',
    'nn_2_0.92013_oof.npy',
    'nn_4_0.92096_oof.npy',
    'nn_5_0.92286_oof.npy',
    'nn_1_0.92266_oof.npy',
    'nn_5_0.92241_oof.npy',
    'nn_5_0.92289_oof.npy',
    'nn_0_0.92191_oof.npy',
    'nn_1_0.92217_oof.npy'
]

class LoadPredictions(GenericDataPreprocessor):
    def __init__(self):
        pass

    def fit_transform(self, X, y=None):
        global my_best, new_best
        preds_my = [os.path.join('predictions', 'train', x)
            for x in my_best]
        preds_new = [os.path.join('predictions', 'new', x)
            for x in new_best]
        preds = preds_my + preds_new

        for i, p in enumerate(preds):
            if '.csv' in p:
                target = pd.read_csv(p).target
            else:
                target = np.load(p)

            X['preds_{}'.format(i)] = target

        return X

    def transform(self, X):
        global my_best, new_best
        preds_my = [os.path.join('predictions', 'test', x)
            for x in my_best]
        preds_new = [os.path.join('predictions', 'new', x)
            for x in new_best]
        preds_new = [x.replace('oof', 'test') for x in preds_new]

        preds = preds_my + preds_new

        for i, p in enumerate(preds):
            if '.csv' in p:
                target = pd.read_csv(p).target
            else:
                target = np.load(p)

            X['preds_{}'.format(i)] = target

        return X

class DropColumns(GenericDataPreprocessor):
    def __init__(self):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        global my_best, new_best
        num_features = len(my_best) + len(new_best)
        preds_cols = ['preds_{}'.format(x) for x in range(num_features)]
        return X[preds_cols]

dl_params = {
    'target': "target",
    'id': "ID_code"
}
data_loader = DataLoader(args.data, **dl_params)
data_loader.preprocess(LoadPredictions, DropColumns, ToNumpy)
data_loader.generate_split(StratifiedKFold,
    n_splits=20, shuffle=True, random_state=42)
## << Read and preprocess data

## >> Create and train model
from catboost import Pool, CatBoostClassifier
from sklearn.metrics import roc_auc_score

model_params = {
    'name':          "stacking_more_0918_20_fold",
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
    'rsm': 0.5, # % of features
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
