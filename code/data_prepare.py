import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.stats import skew, boxcox


def read_file(filename):
    return pd.read_csv(filename)


def transfer_category(col1, col2):
    le_mod = LabelEncoder()
    col = list(set(col1).union(set(col2)))
    le_mod.fit(col)
    col1 = le_mod.transform(col1)
    col2 = le_mod.transform(col2)
    return col1, col2


def transfer_all_cats(df1, df2):
    cats = ['cat{}'.format(i+1) for i in xrange(116)]
    for cat in cats:
        df1[cat], df2[cat] = transfer_category(df1[cat], df2[cat])
    return df1, df2


def fix_skewness(df1, df2):
    df = pd.concat((df1, df2))
    contnames = ['cont{}'.format(i+1) for i in xrange(14)]
    n_train = df1.shape[0]
    for cont in contnames:
        skewness = skew(df[cont])
        if skewness >= 0.25:
            df[cont], _ = boxcox(df[cont] + 1)
    return df[:n_train], df.drop('loss', axis=1)[n_train:]


if __name__ == '__main__':
    data_path = '../data/'
    file_train = 'train.csv'
    file_test = 'test.csv'
    print 'loading file'
    train = read_file(data_path+file_train)
    test = read_file(data_path+file_test)
    print 'transfering category'
    train, test = transfer_all_cats(train, test)
    print 'saving file'
    train.to_csv(data_path+'train_new.csv', index=False)
    test.to_csv(data_path+'test_new.csv', index=False)
    print 'fixing skewness'
    train, test = fix_skewness(train, test)
    print 'saving file'
    train.to_csv(data_path+'train_skew.csv', index=False)
    test.to_csv(data_path+'test_skew.csv', index=False)
