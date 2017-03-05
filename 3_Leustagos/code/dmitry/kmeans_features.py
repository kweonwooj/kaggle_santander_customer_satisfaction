import os
from sklearn.cluster import KMeans
from santander_preprocess import *

def kmeans_features(LOG):
    LOG.info('*'*50)
    LOG.info('# kmeans_features..')
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
    train, test = normalize_features(train, test)

    flist = [x for x in train.columns if not x in ['ID','TARGET']]

    LOG.info('# Adding K-means features..')
    flist_kmeans = []
    for ncl in range(2, 11):
        cls = KMeans(n_clusters=ncl)
        cls.fit_predict(train[flist].values)
        train['kmeans_cluster' + str(ncl)] = cls.predict(train[flist].values)
        test['kmeans_cluster' + str(ncl)] = cls.predict(test[flist].values)
        flist_kmeans.append('kmeans_cluster' + str(ncl))

    LOG.info('# Saving K-means data..')
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    train[['ID'] + flist_kmeans].append(test[['ID'] + flist_kmeans], ignore_index=True).to_csv(
        OUTPUT_PATH + 'kmeans_feats.csv', index=False)
