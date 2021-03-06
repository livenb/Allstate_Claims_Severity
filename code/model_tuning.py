import matplotlib
matplotlib.use('agg')
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import make_scorer, mean_absolute_error
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import xgboost as xgb
import tqdm
import time


def read_data(filename):
    df = pd.read_csv(filename)
    X = df.drop(['id', 'loss'], axis=1).values
    y = df['loss'].values
    features = df.drop(['id', 'loss'], axis=1).columns
    return X, y, features


def MAE(y_true, y_pred):
    '''return negtive mean absolute error, since grid/random search
    are choose to maxize the criterion'''
    error = abs(y_true - y_pred).sum()
    n = len(y_true) * 1.0
    return -error/n


def run_cv_rf(X, y, fold_num=10, **kwargs):
    kf = KFold(fold_num, shuffle=True)
    oob_scores = []
    test_scores = []
    fea_imps = []
    print 'start {}-fold cv'.format(fold_num)
    bar = tqdm.tqdm(total=fold_num)
    for idx_tr, idx_te in kf.split(X):
        rf = RandomForestRegressor(n_estimators=1000, criterion='mse',
                                   max_depth=10, min_samples_split=10,
                                   min_samples_leaf=5, max_features='auto',
                                   max_leaf_nodes=None,
                                   min_impurity_split=1e-07,
                                   bootstrap=True, oob_score=True, n_jobs=-1,
                                   random_state=None, verbose=0,
                                   warm_start=False)
        rf.fit(X[idx_tr], y[idx_tr])
        oob = rf.oob_score_
        y_pred = rf.predict(X[idx_te])
        test_score = mean_absolute_error(y[idx_te], y_pred)
        print 'oob score: {}\t cv score: {}'.format(oob, test_score)
        oob_scores.append(oob)
        test_scores.append(test_score)
        fea_imps.append(rf.feature_importances_)
        bar.update(1)
    idx = np.argmax(test_score)
    fea_imp = fea_imps[idx]
    avg_oob = sum(oob_scores) / fold_num
    avg_cv = sum(test_scores) / fold_num
    print 'avg oob score: {}\t avg cv score: {}\n'.format(avg_oob, avg_cv)
    return fea_imp


def write_log(filename, report):
    idx = np.argsort(report['mean_test_score'])[::-1]
    with open(filename, 'a+') as f:
        f.write('\n'+'*'*70+'\n')
        f.write(time.strftime("%Y/%m/%d - %H:%M:%S"))
        f.write('\n' + '-'*30 + '\n')
        for i in idx:
            pm = report['params'][i]
            mts = report['mean_test_score'][i]
            mtrs = report['mean_train_score'][i]
            rk = report['rank_test_score'][i]
            f.write('parameters: {}\n\n'.format(pm))
            f.write('train score:{}\ttest score:{}\trank:{}\n'.format(-mtrs, -mts, rk))
            f.write('\n' + '-'*30 + '\n')


def plot_errors(train_errors, test_erros, save=False):
    idx = np.argsort(train_errors)
    plt.figure()
    plt.title('Mean Abosulte Error')
    plt.plot(-train_errors[idx], label='train')
    plt.plot(-test_erros[idx], label='test')
    plt.legend(loc='best')
    if save:
        time_now = time.strftime("%Y%m%d_%H%M%S")
        name = '../img/erros_{}.png'.format(time_now)
        plt.savefig(name, dpi=300)


def plot_feature_importance(fea_imp, features, fea_num=None, filename=None):
    if fea_num:
        k = fea_num
    else:
        k = len(fea_imp)
    idx = fea_imp.argsort()[::-1]
    plt.figure()
    plt.title('Feature Importance')
    plt.bar(range(k), fea_imp[idx][:k], align="center", alpha=0.8)
    plt.xticks(range(k), features[idx][:k], rotation=60)
    if filename:
        plt.savefig('../img/fea_imp.png', dpi=300)


def rf_params_search(X, y):
    params = {'max_features': ['sqrt'],
              'max_depth': stats.randint(20, 35),
              'min_samples_split': stats.randint(1, 51),
              'min_samples_leaf': stats.randint(1, 21)}
    rfrgs = RandomForestRegressor(n_estimators=500,
                                  criterion='mse', oob_score=True)
    n_iter_search = 80
    random_search = RandomizedSearchCV(rfrgs, param_distributions=params,
                                       n_iter=n_iter_search,
                                       cv=5, n_jobs=-1, verbose=1,
                                       scoring=make_scorer(MAE)
                                       )
    random_search.fit(X, y)
    print 'finish training'
    report = random_search.cv_results_
    write_log('../log/random_forest_params.log', report)
    train_errors = report['mean_train_score']
    test_errors = report['mean_test_score']
    plot_errors(train_errors, test_errors, True)
    return random_search


def xgb_params_search(X, y):
    params = {'learning_rate': [0.1],
              'gamma': stats.uniform(0, 11),
              'max_depth': stats.randint(3, 21),
              'min_child_weight': stats.randint(1, 31),
              'subsample': [1],
              'colsample_bytree': [1],
              'colsample_bylevel': [1],
              }
    xgb_rgs = XGBRegressor(n_estimators=500, objective='reg:linear', nthread=4)
    n_iter_search = 40
    random_search = RandomizedSearchCV(xgb_rgs, param_distributions=params,
                                       n_iter=n_iter_search,
                                       cv=5, n_jobs=4, verbose=1,
                                       scoring=make_scorer(MAE)
                                       )
    random_search.fit(X, y)
    print 'finish training'
    report = random_search.cv_results_
    write_log('../log/xgboost_params.log', report)
    train_errors = report['mean_train_score']
    test_errors = report['mean_test_score']
    plot_errors(train_errors, test_errors, True)
    return random_search


def run_grid_search(X, y):
    params = {'learning_rate': [0.001, 0.01, 0.05, 0.075, 0.1, 0.2, 0.3],
              'gamma': [0],
              'max_depth': [6],
              'min_child_weight': [1],
              'subsample': [1],
              'colsample_bytree': [0.5],
              'colsample_bylevel': [0.6],
              }
    xgb_rgs = XGBRegressor(n_estimators=1500, objective='reg:linear', nthread=4)
    grid_search = GridSearchCV(xgb_rgs, param_grid=params,
                               cv=5, n_jobs=4, verbose=1,
                               scoring=make_scorer(MAE)
                               )
    grid_search.fit(X, y)
    print 'finish training'
    report = grid_search.cv_results_
    write_log('../log/xgboost_params.log', report)
    train_errors = report['mean_train_score']
    test_errors = report['mean_test_score']
    plot_errors(train_errors, test_errors, True)
    return grid_search


if __name__ == '__main__':
    data_path = '../data/'
    file_train = 'train_new.csv'
    X, y, features = read_data(data_path+file_train)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # run_grid_search(X, y)
    # fea_imp = run_cv_rf(X_train, y_train)
    random_search = xgb_params_search(X, y)
    # fea_imp = random_search.best_estimator_.feature_importances_
    # plot_feature_importance(fea_imp, features, 30, True)
    # plt.show()
