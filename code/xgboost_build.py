import pandas as pd
import xgboost as xgb


def one_run_xgboost(X, y):
    xgdmat = xgb.DMatrix(X, y)
    params = {'eta': 0.0117, 'subsample': 0.7,
              'colsample_bylevel': 0.95, 'gamma': 3.75,
              'colsample_bytree': 0.85, 'objective': 'reg:linear',
              'max_depth': 13, 'min_child_weight': 26}
    num_rounds = 1000
    bst = xgb.train(params, xgdmat, num_boost_round=num_rounds)
    return bst


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


if __name__ == '__main__':
    data_path = '../data/'
    file_train = 'train_new.csv'
    file_test = 'test_new.csv'
    X, y, features = read_data(data_path+file_train)
    X_test, ids = read_test(data_path+file_test)
    X_test = xgb.DMatrix(X_test)
    # bagging, y_pred = build_random_forest(X, y, X_test)
    bst = one_run_xgboost(X, y)
    y_pred = bst.predict(X_test)
    make_submission(y_pred, ids, '../data/rf_res.csv')
