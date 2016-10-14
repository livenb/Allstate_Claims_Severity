import pandas as pd
import numpy as np
from scipy import stats 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.metrics import make_scorer, mean_absolute_error
import matplotlib.pyplot as plt
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
        f.write('\n'+'*'*30+'\n')
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


def rf_params_search(X, y):
    params = {'max_features': ['sqrt', 'log2'],
              'max_depth': stats.randint(5, 35),
              'min_samples_split': stats.randint(1, 51),
              'min_samples_leaf': stats.randint(1, 51)}
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
    plot_errors(train_errors, test_errors)
    return random_search


def one_run_xgboost(X, y):
    xgdmat = xgb.DMatrix(X, y)
    params = {'eta': 0.01, 'seed': 0, 'subsample': 0.5,
              'colsample_bytree': 0.5, 'objective': 'reg:linear',
              'max_depth': 6, 'min_child_weight': 3}
    num_rounds = 1000
    bst = xgb.train(params, xgdmat, num_boost_round=num_rounds)


def plot_errors(train_errors, test_erros):
    idx = np.argsort(train_errors)
    plt.figure()
    plt.title('Mean Abosulte Error')
    plt.plot(-train_errors[idx], label='train')
    plt.plot(-test_erros[idx], label='test')
    plt.legend()


def plot_feature_importance(fea_imp, features, fold_num=None, filename=None):
    if fold_num:
        k = fold_num
    else:
        k = len(fea_imp)
    idx = fea_imp.argsort()[::-1]
    plt.figure()
    plt.title('Feature Importance')
    plt.bar(range(k), fea_imp[idx][:k], align="center")
    plt.xticks(range(k), features[idx][:k], rotation=60)
    if filename:
        plt.savefig('../img/fea_imp_simple.png', dpi=300)


def run_grid_search(X, y):
    pass

if __name__ == '__main__':
    data_path = '../data/'
    file_train = 'train_new.csv'
    X, y, features = read_data(data_path+file_train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # fea_imp = run_cv_rf(X_train, y_train)
    # plot_feature_importance(fea_imp, features)
    random_search = rf_params_search(X_train, y_train)
    plt.show()
