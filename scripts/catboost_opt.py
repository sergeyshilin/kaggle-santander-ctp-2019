import numpy as np
import pandas as pd
import os

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
counters = np.load('../data/value_counters.npy')

test.drop('ID_code', inplace=True, axis=1)

from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

X_tr, y_tr = train[train.columns[2:]], train.target
generator = skf.split(X_tr, y_tr)
tr_ids, cv_ids = next(generator)
X_cv, y_cv = X_tr.iloc[cv_ids], y_tr.iloc[cv_ids]
X_tr, y_tr = X_tr.iloc[tr_ids], y_tr.iloc[tr_ids]


class UpsamplingPreprocessor:
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

class CountersGeneratorOnline:
    def __init__(self):
        self.counters = np.load(os.path.join(
            '../data/', 'value_counters.npy'))

    def fit_transform(self, X, y=None):
        return self.transform(X), y

    def transform(self, X):
        var_cols = ['var_{}'.format(x) for x in range(200)]

        for i, c in enumerate(var_cols):
            X['varcount_' + c] = X[c].map(self.counters[i]).astype(np.int32)
            X['unique_' + c] = (X['varcount_' + c] == 1)
            X['varcount_' + c] = X['varcount_' + c] * X[c]

        return X

augmenter = UpsamplingPreprocessor()
counters_generator = CountersGeneratorOnline()

X_tr, y_tr = augmenter.fit_transform(X_tr, y_tr)
X_cv = augmenter.transform(X_cv)

X_tr, y_tr = counters_generator.fit_transform(X_tr, y_tr)
X_cv = counters_generator.transform(X_cv)


from bayes_opt import BayesianOptimization
import warnings
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from sklearn.metrics import roc_auc_score
import xgboost as xgb


start_params = {
    'loss_function': "Logloss",
    'eval_metric': "AUC",
    'task_type': "GPU",
    'learning_rate': 0.03,
    'iterations': 20000,
    'random_seed': 42,
    'od_type': "Iter",
    'early_stopping_rounds': 300,
}

def evaluate_cb(**params):
    print('='*100)
    warnings.simplefilter('ignore')
    params['depth'] = int(params['depth'])
    params['border_count'] = int(params['border_count'])

    start_params.update(params)
    print('Training with params: {}'.format(params))
    clf = cb.CatBoostClassifier(**start_params)
    clf.fit(trn_data, eval_set = val_data, use_best_model = True,
            silent=True, early_stopping_rounds=300)
    val_pred = clf.predict_proba(val_data)[:,1]
    val_score = roc_auc_score(y_cv, val_pred)
    print("Model trees: {}; Val score: {:<8.5f}".format(clf.tree_count_,val_score))
    return val_score

bounds = {
    'l2_leaf_reg': (0.5, 100.0),
    'bagging_temperature': (0.1, 100.0),
    'random_strength': (0, 20),
    'depth': (2, 6),
    'border_count': (32, 128)
}

bo = BayesianOptimization(evaluate_cb, pbounds=bounds,random_state=42)
logger = JSONLogger(path="./catboost_bo.json")
bo.subscribe(Events.OPTMIZATION_STEP, logger)

bo.probe(
    params={
        'l2_leaf_reg': 3.0,
        'bagging_temperature': 1,
        'random_strength': 1,
        'depth': 3,
        'border_count':128
    },
    lazy=True,
)

bo.maximize(init_points=20, n_iter=300)
print (bo.max)
