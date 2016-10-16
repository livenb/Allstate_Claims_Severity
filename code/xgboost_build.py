import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor


def read_data(filename):
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


def make_submission(y_pred, ids, filename):
    df = pd.DataFrame()
    df['id'] = ids
    df['loss'] = y_pred
    df.to_csv(filename, index=False)


def one_run_xgboost(X, y):
    xgdmat = xgb.DMatrix(X, y)
    params = {'colsample_bytree': 0.8, 'colsample_bylevel': 0.85,
              'learning_rate': 0.025, 'min_child_weight': 29,
              'subsample': 0.95, 'max_depth': 12, 'gamma': 0.079}
    num_rounds = 1000
    bst = xgb.train(params, xgdmat, num_boost_round=num_rounds)
    return bst


def bagging_xgboost(X, y, X_test):
    params = {'colsample_bytree': 0.7, 'colsample_bylevel': 1,
              'learning_rate': 0.075, 'min_child_weight': 6,
              'subsample': 0.7, 'max_depth': 6, 'gamma': 0}
    xgb_rgs = XGBRegressor(n_estimators=1000, nthread=4, **params)
    bagging = BaggingRegressor(xgb_rgs, n_estimators=10, n_jobs=2, verbose=2)
    bagging.fit(X, y)
    y_pred = bagging.predict(X_test)
    return bagging, y_pred


if __name__ == '__main__':
    data_path = '../data/'
    file_train = 'train_new.csv'
    file_test = 'test_new.csv'
    X, y, features = read_data(data_path+file_train)
    X_test, ids = read_test(data_path+file_test)
    # X_test = xgb.DMatrix(X_test)
    bagging, y_pred = bagging_xgboost(X, y, X_test)
    # bst = one_run_xgboost(X, y)
    # y_pred = bst.predict(X_test)
    make_submission(y_pred, ids, '../data/rf_res.csv')
