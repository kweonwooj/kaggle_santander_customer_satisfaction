import numpy as np
import os
np.random.seed(12324)
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from tsne import bh_sne
from santander_preprocess import *

def tsne_features(LOG):
    LOG.info('*'*50)
    LOG.info('# tsne_features..')
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

    LOG.info('# Adding TSNE features..')
    X = train[flist].append(test[flist], ignore_index=True).values.astype('float64')
    svd = TruncatedSVD(n_components=30)
    X_svd = svd.fit_transform(X)
    X_scaled = StandardScaler().fit_transform(X_svd)

    LOG.info('# Fitting TSNE..')
    feats_tsne = bh_sne(X_scaled)
    feats_tsne = pd.DataFrame(feats_tsne, columns=['tsne1', 'tsne2'])
    feats_tsne['ID'] = train[['ID']].append(test[['ID']], ignore_index=True)['ID'].values
    train = pd.merge(train, feats_tsne, on='ID', how='left')
    test = pd.merge(test, feats_tsne, on='ID', how='left')

    LOG.info('# Saving TSNE data..')
    feat = train[['ID', 'tsne1', 'tsne2']].append(test[['ID', 'tsne1', 'tsne2']], ignore_index=True)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    feat.to_csv(OUTPUT_PATH + 'tsne_feats.csv', index=False)
