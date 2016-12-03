import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from datetime import datetime


def batch_generator(X, y, batch_size, shuffle):
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index, :].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0


def batch_generator_pred(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size*counter: batch_size*(counter+1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0


def shuffle_data(data):
    idx = list(data.index)
    np.random.shuffle(idx)
    data = data.iloc[idx]
    return data


def data_preprocessing(train, test):
    cat_cols = ['cat{}'.format(i+1) for i in xrange(116)]
    num_cols = ['cont{}'.format(i+1) for i in xrange(14)]

    train = shuffle_data(train)

    y = np.log(train['loss'].values+200)
    id_test = test['id'].values
    ntrain = train.shape[0]
    tr_te = pd.concat((train, test), axis=0)
    sparse_data = []

    for col in cat_cols:
        dummy = pd.get_dummies(tr_te[col].astype('category'))
        tmp = csr_matrix(dummy)
        sparse_data.append(tmp)

    scaler = StandardScaler()
    tmp = csr_matrix(scaler.fit_transform(tr_te[num_cols]))
    sparse_data.append(tmp)

    del(tr_te, train, test)
    xtr_te = hstack(sparse_data, format='csr')
    xtrain = xtr_te[:ntrain, :]
    xtest = xtr_te[ntrain:, :]

    del(xtr_te, sparse_data, tmp)
    return xtrain, y, xtest, id_test


def nn_model(dim):
    model = Sequential()

    model.add(Dense(400, input_dim=dim, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(200, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(50, init='he_normal'))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(1, init='he_normal'))
    model.compile(loss='mae', optimizer='adadelta')
    return model


def run_nfold_nn(train_X, train_y, test_X, nfolds=10, nbags=10):
    folds = KFold(len(train_y), n_folds=nfolds, shuffle=True, random_state=111)
    nepochs = 55
    pred_oob = np.zeros(train_X.shape[0])
    pred_test = np.zeros(test_X.shape[0])

    for i, (inTr, inTe) in enumerate(folds):
        xtr = train_X[inTr]
        ytr = train_y[inTr]
        xte = train_X[inTe]
        yte = train_y[inTe]
        pred = np.zeros(xte.shape[0])
        for j in range(nbags):
            print '-'*30
            print '\nFold %d Bag %d training...' % (i+1, j+1)
            print '-'*30
            model = nn_model(train_X.shape[0])
            model.fit_generator(generator=batch_generator(xtr, ytr, 128, True),
                                nb_epoch=nepochs,
                                samples_per_epoch=xtr.shape[0],
                                verbose=1)
            pred_temp = model.predict_generator(
                                            generator=batch_generator_pred(xte, 800, False),
                                            val_samples=xte.shape[0])
            pred += np.exp(pred_temp[:, 0])-200
            pred_tp_te = model.predict_generator(
                                                generator=batch_generator_pred(test_X, 800, False),
                                                val_samples=test_X.shape[0])
            pred_test += np.exp(pred_tp_te[:, 0])-200
        pred /= nbags
        pred_oob[inTe] = pred
        score = mean_absolute_error(np.exp(yte)-200, pred)
        i += 1
        print 'Fold: {} \t-MAE: {}'.format(i, score)
    avg_score = mean_absolute_error(np.exp(train_y)-200, pred_oob)
    print 'Total - MAE:', avg_score
    pred_test /= (nfolds*nbags)
    return pred_test, avg_score


if __name__ == '__main__':
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')
    train_X, train_y, test_X, id_test = data_preprocessing(train, test)
    nfolds, nbags = 10, 10
    pred_test, mae = run_nfold_nn(train_X, train_y, test_X,
                                  nfolds=nfolds, nbags=nbags)
    res = pd.DataFrame({'id': id_test, 'loss': pred_test})
    outname = '../data/submission__nn_{}fold_{}bags_{}_{}.csv'.format(nfolds, nbags, mae, datetime.now().strftime("%Y-%m-%d-%H-%M"))
    res.to_csv('../data/submission__nn.csv', index=False)
