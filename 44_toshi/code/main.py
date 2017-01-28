'''
    This file is python implementation of toshi_k's solution in Lua, R.

    Original Link: https://github.com/toshi-k/kaggle-santander-customer-satisfaction
'''

import os
import time
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from utils.data_utils import *
from utils.model_utils import *
from utils.feature_utils import *

import numpy as np
import pandas as pd
import scipy
np.random.seed(777)


def main():

    print('=' * 50)
    print('# Santander Customer Satisfaction _ 44_toshi')
    print('-' * 50)

    # preprocess data
    prepare_data()

    # generate derivative feature
    generate_name_feature()

    ##################################################################################################################
    ### Loading data
    ##################################################################################################################

    print('=' * 50)
    print('# Loading data..')
    print('-' * 50)

    trn = pd.read_csv('./input/train.csv')
    tst = pd.read_csv('./input/test.csv')

    add_trn = pd.read_csv('./input/name_feat_trn.csv')
    add_tst = pd.read_csv('./input/name_feat_tst.csv')

    trn = trn.merge(add_trn, on='ID')
    tst = tst.merge(add_tst, on='ID')

    trn.fillna(-999999, inplace=True)
    tst.fillna(-999999, inplace=True)

    y = trn['TARGET'].values
    test_id = trn['ID'].values

    trn.drop(['ID','TARGET'], axis=1, inplace=True)
    tst.drop(['ID'], axis=1, inplace=True)

    ##################################################################################################################
    ### Cross Validations
    ##################################################################################################################

    print('=' * 50)
    print('# Performing Cross-Validation..')
    print('-' * 50)

    xgb_engine(trn, tst, y, test_id)


if __name__ == '__main__':
    start = time.time()
    main()
    print('# Finished ({:.2f} sec elapsed)'.format(time.time() - start))

