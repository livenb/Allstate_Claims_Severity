import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import make_scorer, mean_absolute_error
import matplotlib.pyplot as plt
import time


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


def build_random_forest(X, y, X_test):
    rfr = RandomForestRegressor(n_estimators=2000, criterion='mse',
                                max_depth=33, max_features='sqrt',
                                min_samples_split=16, min_samples_leaf=2,
                                oob_score=True, verbose=1,
                                n_jobs=-1)
    bagging = BaggingRegressor(rfr, n_estimators=10, verbose=1)
    bagging.fit(X, y)
    y_pred = bagging.predict(X_test)
    return bagging, y_pred


def make_submission(y_pred, ids, filename):
    df = pd.DataFrame()
    df['id'] = ids
    df['loss'] = y_pred
    df.to_csv(filename, index=False)


if __name__ == '__main__':
    data_path = '../data/'
    file_train = 'train_new.csv'
    file_test = 'test_new.csv'
    X, y, features = read_data(data_path+file_train)
    X_test, ids = read_test(data_path+file_test)
    bagging, y_pred = build_random_forest(X, y, X_test)
    make_submission(y_pred, ids, '../data/rf_res.csv')
