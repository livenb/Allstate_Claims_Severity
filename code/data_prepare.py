import pandas as pd
from sklearn.preprocessing import LabelEncoder


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
