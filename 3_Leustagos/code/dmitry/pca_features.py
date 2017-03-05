import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from santander_preprocess import *

def pca_features(LOG):
    LOG.info('*'*50)
    LOG.info('# pca_features..')
    # input & output path redefined
    INPUT_PATH = '../input/'
    OUTPUT_PATH = './feats/'

    LOG.info('# Loading data..')
    train = pd.read_csv(INPUT_PATH + 'train.csv')
    test = pd.read_csv(INPUT_PATH + 'test.csv')

    LOG.info('# Preprocessing..')
    train, test = process_base(train, test)
    train, test = drop_sparse(train, test)
    train, test = drop_duplicated(train, test)
    train, test = add_features(train, test, ['SumZeros'])

    flist = [x for x in train.columns if not x in ['ID','TARGET']]

    LOG.info('# Adding PCA features..')
    pca = PCA(n_components=2)
    x_train_projected = pca.fit_transform(normalize(train[flist], axis=0))
    x_test_projected = pca.transform(normalize(test[flist], axis=0))
    train.insert(1, 'PCAOne', x_train_projected[:, 0])
    train.insert(1, 'PCATwo', x_train_projected[:, 1])
    test.insert(1, 'PCAOne', x_test_projected[:, 0])
    test.insert(1, 'PCATwo', x_test_projected[:, 1])

    LOG.info('# Saving PCA data..')
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    pca_feats = train[['ID', 'PCAOne', 'PCATwo']].append(test[['ID', 'PCAOne', 'PCATwo']], ignore_index=True)
    pca_feats.to_csv(OUTPUT_PATH + 'dmitry_pca_feats.csv')
