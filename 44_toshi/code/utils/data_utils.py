
"""
    This file has a utility function for data IO
"""

import pandas as pd
import numpy as np
import itertools
import os

def prepare_data(LOG):

    LOG.info('# Load data')
    train_path = '../input/train.csv'
    test_path  = '../input/test.csv'

    trn = pd.read_csv(train_path)
    tst = pd.read_csv(test_path)

    trg = trn['TARGET']
    trn.drop(['TARGET'], axis=1, inplace=True)

    data = pd.concat([trn, tst], axis=0)
    data.reset_index(drop=True)

    LOG.info('# Remove constants')
    constants = []
    for col in data.columns:
        if len(np.unique(data[col])) == 1:
            constants.append(col)
    data.drop(constants, axis=1, inplace=True)

    LOG.info('# Remove duplicates')
    pairs = list(itertools.combinations(data.columns, 2))
    to_remove = []
    for f1, f2 in pairs:
        if f1 not in to_remove and f1 not in to_remove:
            if (data[f1].equals(data[f2])):
                to_remove.append(f2)
    data.drop(to_remove, axis=1, inplace=True)

    LOG.info('# Replace NAs')
    data[data == -999999] = np.nan
    data[data == 9999999999] = np.nan

    trn_2 = data.iloc[:trn.shape[0], :]
    tst_2 = data.iloc[trn.shape[0]:, :]
    trn_2['TARGET'] = trg

    if not os.path.exists('./input'):
        os.mkdir('./input')

    LOG.info('# Save to files')
    trn_2.to_csv('./input/train.csv', index=False)
    tst_2.to_csv('./input/test.csv', index=False)
