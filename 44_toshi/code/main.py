'''
    This file is python implementation of toshi_k's solution in Lua, R.

    Original Link: https://github.com/toshi-k/kaggle-santander-customer-satisfaction
'''

import time

from utils.data_utils import *
from utils.xgb_utils import *
from utils.ann_utils import *
from utils.feature_utils import *
from utils.log_utils import *

LOG = get_logger('44toshi_solution.log')

import numpy as np
import pandas as pd


np.random.seed(777)


def main():

    LOG.info('=' * 50)
    LOG.info('# Santander Customer Satisfaction _ 44_toshi')
    LOG.info('-' * 50)

    # preprocess data
    prepare_data(LOG)

    # generate derivative feature
    generate_name_feature(LOG)

    ##################################################################################################################
    ### Loading data
    ##################################################################################################################

    LOG.info('=' * 50)
    LOG.info('# Loading data..')
    LOG.info('-' * 50)

    trn = pd.read_csv('./input/train.csv')
    tst = pd.read_csv('./input/test.csv')

    add_trn = pd.read_csv('./input/name_feature_train.csv')
    add_tst = pd.read_csv('./input/name_feature_test.csv')

    trn = trn.merge(add_trn, on='ID')
    tst = tst.merge(add_tst, on='ID')

    trn.fillna(-999999, inplace=True)
    tst.fillna(-999999, inplace=True)

    y = trn['TARGET'].values
    test_id = tst['ID'].values

    trn.drop(['ID', 'TARGET'], axis=1, inplace=True)
    tst.drop(['ID'], axis=1, inplace=True)

    ##################################################################################################################
    ### Cross Validations
    ##################################################################################################################

    LOG.info('=' * 50)
    LOG.info('# Performing Cross-Validation..')
    LOG.info('-' * 50)

    xgb_engine(trn, tst, y, test_id, LOG)
    #ann_engine(trn, tst, y, test_id, LOG)

    ##################################################################################################################
    ### Ensemble _ Averaging
    ##################################################################################################################

    xgb_ensemble()
    #ann_ensemble()

if __name__ == '__main__':
    start = time.time()
    main()
    LOG.info('=' * 50)
    LOG.info('# Finished ({:.2f} sec elapsed)'.format(time.time() - start))
    LOG.info('=' * 50)
