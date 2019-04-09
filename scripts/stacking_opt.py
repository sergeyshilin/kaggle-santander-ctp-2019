import numpy as np
import pandas as pd
import os

train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')
# counters = np.load('../data/value_counters.npy')

test.drop('ID_code', inplace=True, axis=1)
X_tr, y_tr = train[train.columns[2:]], train.target


my_best = [
    '9128_catboost_true_counters_2503133840.csv',
    '9128_xgboost_counters_10_folds_2603205754.csv',
    '9211_xgboost_multiplied_counter_10_folds_0604015413.csv',
    '9202_separate_feature_models_threshold_0.84_10fold_04042353.csv',
    '9223_catboost_unique_10_fold_20x_upsample_0104001611.csv',
    '9208_separate_feature_models_threshold_0.80_10fold_07041509.csv',
    '9222_catboost_multiplied_counter_10_folds_0504034846.csv',
    '9118_xgboost_with_counters_2603202144.csv',
    '9133_catboost_ranks_is_unique_0104154327.csv',
    '9139_catboost_counters_10_folds_2603222240.csv',
    '9111_lightgbm_with_counters_2603170158.csv',
    '9215_catboost_unique_10x_upsample_3_balancer_0204145755.csv',
    '9211_catboost_unique_with_10x_upsample_3103172847.csv',
    '9199_xgboost_unique_10x_upsample_0104062006.csv',
    '9222_catboost_poisson_bootstrap_10_folds_0804042543.csv',
    '9063_catboost_with_counts_2503021850.csv',
    '9118_lightgbm_with_counters_2603173935.csv',
    '9218_catboost_excluded_features_10_folds_0804184027.csv',
    '9211_catboost_unbalanced_weights_0304111611.csv',
    '9219_catboost_best_parameters_10_folds_0704212230.csv',
    '9212_catboost_unique_10x_upsample_5_balancer_0204172825.csv',
    '9197_lightgbm_unique_10x_upsample_0104035943.csv',
    '9210_lightgbm_multiplied_counter_10_folds_0604220718.csv',
    '9218_catboost_multiplied_counter_0304011056.csv',
    '9132_catboost_is_unique_3003140136.csv'
]

new_best = [
    'nn_0_0.92345_oof.npy',
    'nn_aug_0.92303_12_oof.npy',
    'nn_3_0.92114_oof.npy',
    'nn_3_0.92235_oof.npy',
    'lgb_0.91145_oof.npy',
    'nn_4_0.92292_oof.npy',
    'nn_0_0.92269_oof.npy',
    'cat_0.91617_oof.npy',
    'nn_3_0.92287_oof.npy',
    'nn_5_0.92258_oof.npy',
    'nn_1_0.92294_oof.npy',
    '!cat_0.91558_oof.npy',
    'nn_2_0.92128_oof.npy',
    'nn_1_0.92273_oof.npy',
    'nn_0_0.92226_oof.npy',
    'lgb_0.91305_oof.npy',
    'nn_2_0.92013_oof.npy',
    'nn_4_0.92096_oof.npy',
    'xgb_0.91221_oof.npy',
    'nn_5_0.92286_oof.npy',
    'nn_1_0.92266_oof.npy',
    'nn_5_0.92241_oof.npy',
    'nn_5_0.92289_oof.npy',
    'nn_0_0.92191_oof.npy',
    'nn_1_0.92217_oof.npy'
]

class LoadPredictions:
    def __init__(self):
        pass

    def fit_transform(self, X, y=None):
        global my_best, new_best
        preds_my = [os.path.join('../predictions', 'train', x)
            for x in my_best]
        preds_new = [os.path.join('../predictions', 'new', x)
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
        preds_my = [os.path.join('../predictions', 'test', x)
            for x in my_best]
        preds_new = [os.path.join('../predictions', 'new', x)
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

class DropColumns:
    def __init__(self):
        pass

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X):
        global my_best, new_best
        num_features = len(my_best) + len(new_best)
        # var_cols = ['var_{}'.format(x) for x in range(200)]
        preds_cols = ['preds_{}'.format(x) for x in range(num_features)]
        X[preds_cols] = X[preds_cols].rank()
        # X[var_cols] = np.round(X[var_cols], decimals=3)
        return X[preds_cols]

lp = LoadPredictions()
dc = DropColumns()

X_tr = lp.fit_transform(X_tr)
X_te = lp.transform(test)

X_tr = dc.fit_transform(X_tr)
X_te = dc.transform(X_te)


from sklearn.model_selection import StratifiedKFold
from bayes_opt import BayesianOptimization
import warnings
from bayes_opt.observer import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from sklearn.metrics import roc_auc_score
from catboost import Pool, CatBoostClassifier


start_params = {
    'loss_function': "Logloss",
    'eval_metric': "AUC",
    'task_type': "GPU",
    'learning_rate': 0.01,
    'iterations': 25000,
    'random_seed': 42,
    'od_type': "Iter",
    'bootstrap_type': "Poisson",
}

def evaluate_cb(**params):
    print('='*100)
    warnings.simplefilter('ignore')
    params['depth'] = int(params['depth'])
    params['border_count'] = int(params['border_count'])

    start_params.update(params)
    print('Training with params: {}'.format(params))

    oof_preds = np.zeros((len(X_tr), 1))
    skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    for tr_ids, cv_ids in skf.split(X_tr, y_tr):
        train_x, train_y = X_tr.iloc[tr_ids], y_tr.iloc[tr_ids]
        cv_x, cv_y = X_tr.iloc[cv_ids], y_tr.iloc[cv_ids]
        trn_data = Pool(train_x, label=train_y)
        val_data = Pool(cv_x, label=cv_y)

        clf = CatBoostClassifier(**start_params)

        clf.fit(trn_data, eval_set = val_data, use_best_model = True,
                silent=True, early_stopping_rounds=100)
        val_pred = clf.predict_proba(val_data)[:,1]
        oof_preds[cv_ids, :] = val_pred.reshape((-1, 1))

    val_score = roc_auc_score(y_tr, oof_preds)
    print("Model trees: {}; Val score: {:<8.5f}".format(clf.tree_count_,val_score))
    return val_score

bounds = {
    'l2_leaf_reg': (0.5, 100.0),
    'bagging_temperature': (0.1, 100.0),
    'random_strength': (0, 20),
    'subsample': (0.2, 1.0),
    'depth': (1, 3),
    'border_count': (32, 128)
}

bo = BayesianOptimization(evaluate_cb, pbounds=bounds,random_state=42)
logger = JSONLogger(path="./stacking_bo.json")
bo.subscribe(Events.OPTMIZATION_STEP, logger)

bo.probe(
    params={
        'l2_leaf_reg': 3.0,
        'bagging_temperature': 1,
        'random_strength': 1,
        'subsample': 0.3,
        'depth': 2,
        'border_count':128
    },
    lazy=True,
)

bo.maximize(init_points=5, n_iter=10)
print (bo.max)
