import pandas as pd
import xgboost as xgb
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import KFold


def read_train(filename):
    df = pd.read_csv(filename)
    X = df.drop(['id', 'loss'], axis=1).values
    y = df['loss'].values
    features = df.drop(['id', 'loss'], axis=1).columns
    return X, y, features


def read_test(filename):
    df = pd.read_csv(filename)
    X = df.drop('id', axis=1).values
    ids = df['id'].values
    return X, ids


def load_data(trainfile, testfile):
    X, y, features = read_train(trainfile)
    X_test, ids = read_test(testfile)
    return X, y, X_test, features, ids


def make_submission(y_pred, ids, filename):
    df = pd.DataFrame()
    df['id'] = ids
    df['loss'] = y_pred
    df.to_csv(filename, index=False)


def one_run_xgboost(X, y, X_test, params, num_rounds=1000):
    xgdmat = xgb.DMatrix(X, np.log(y))
    X_test = xgb.DMatrix(X_test)
    bst = xgb.train(params, xgdmat, num_boost_round=num_rounds)
    y_pred = bst.predict(X_test)
    return np.exp(y_pred)


def bagging_xgboost(X, y, X_test, params):
    xgb_rgs = XGBRegressor(n_estimators=2000, nthread=4, **params)
    bagging = BaggingRegressor(xgb_rgs, n_estimators=20, n_jobs=2, verbose=2)
    bagging.fit(X, np.log(y))
    y_pred = bagging.predict(X_test)
    return np.exp(y_pred) - 200


def avg_xgboost(X, y, X_test, params, n_fold=5, n_trees=20000, n_early=100):
    params['objective'] = 'reg:linear'
    kfold = KFold(n_fold)
    y_pred = np.zeros(X_test.shape[0])
    X_test = xgb.DMatrix(X_test)
    for i, (idx_train, idx_eval) in enumerate(kfold.split(X)):
        X_tr, y_tr = X[idx_train], y[idx_train]
        X_eval, y_eval = X[idx_eval], y[idx_eval]
        d_tr = xgb.DMatrix(X_tr, np.log(y_tr))
        d_ev = xgb.DMatrix(X_eval, np.log(y_eval))
        watchlist = [(d_ev, 'eval')]
        bst = xgb.train(params, d_tr, num_boost_round=n_trees,
                        evals=watchlist, early_stopping_rounds=n_early)
        y_pred += np.exp(bst.predict(X_test))
    y_pred /= n_fold
    return y_pred - 200


if __name__ == '__main__':
    path = '../data/'
    file_train = 'train_new.csv'
    file_test = 'test_new.csv'
    skew_train = 'train_skew.csv'
    skew_test = 'test_skew.csv'
    # X, y, X_test, features, ids = load_data(path+file_train, path+file_test)
    X, y, X_test, features, ids = load_data(path+skew_train, path+skew_test)
    params = {'colsample_bytree': 0.5, 'colsample_bylevel': 0.6,
              'learning_rate': 0.05, 'min_child_weight': 1,
              'subsample': 1, 'max_depth': 6, 'gamma': 0}
    y_pred_b = bagging_xgboost(X, y, X_test, params)
    make_submission(y_pred_b, ids, '../data/rf_res_1.csv')
    y_pred_a = avg_xgboost(X, y, X_test, params)
    make_submission(y_pred_a, ids, '../data/rf_res_2.csv')
