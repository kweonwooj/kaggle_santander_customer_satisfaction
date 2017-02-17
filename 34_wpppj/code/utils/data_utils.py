
"""
    This file has a utility function for data IO
"""

import pandas as pd
import numpy as np
import itertools
import os

def prepare_data(LOG):

    LOG.info('# Load data')

    trn_raw = pd.read_csv('../input/train.csv')
    tst_raw = pd.read_csv('../input/test.csv')
    trn_y = trn_raw[['ID', 'TARGET']]
    trn_x = trn_raw.copy()
    tst_x = tst_raw.copy()
    tst_x['TARGET'] = -1
    tst_y = pd.DataFrame({'ID': tst_x['ID'], 'TARGET': tst_x['TARGET']})

    print('# Remove constants')
    constants = []
    for col in trn_x.columns:
        if len(np.unique(trn_x[col])) == 1:
            constants.append(col)
    trn_x.drop(constants, axis=1, inplace=True)
    tst_x.drop(constants, axis=1, inplace=True)

    print('# Remove duplicates')
    pairs = list(itertools.combinations(trn_x.columns, 2))
    to_remove = []
    for f1, f2 in pairs:
        if f1 not in to_remove and f1 not in to_remove:
            if (trn_x[f1].equals(trn_x[f2])):
                to_remove.append(f2)
    trn_x.drop(to_remove, axis=1, inplace=True)
    tst_x.drop(to_remove, axis=1, inplace=True)

    print('# Convert extreme values')
    trn_x[trn_x > 9999999990] = -1
    trn_x[trn_x < -99999] = -1

    print('# Remove sparse features')
    sparse_features = []
    for col in trn_x.columns:
        if trn_x[col].quantile(0.999) == 0:
            sparse_features.append(col)
    trn_x.drop(sparse_features, axis=1, inplace=True)
    tst_x.drop(sparse_features, axis=1, inplace=True)

    print('# Transform columns')
    transform_cols = ['num_meses_var8_ult3', 'num_meses_var13_largo_ult3', 'num_op_var40_comer_ult1']
    for col in transform_cols:
        if col.split('_')[1] == 'meses':
            trn_x[col + '_trans'] = (trn_x[col] > 0).astype(int)
            tst_x[col + '_trans'] = (tst_x[col] > 0).astype(int)
        else:
            trn_x[col + '_trans'] = trn_x[col] / 3
            tst_x[col + '_trans'] = tst_x[col] / 3
    trn_x.drop(transform_cols, axis=1, inplace=True)
    tst_x.drop(transform_cols, axis=1, inplace=True)

    if not os.path.exists('./input'):
        os.mkdir('./input')

    LOG.info('# Save to files')
    trn_x.to_csv('./input/trn_x.csv', index=False)
    tst_x.to_csv('./input/tst_x.csv', index=False)
    print('# DONE : prepare_data')

