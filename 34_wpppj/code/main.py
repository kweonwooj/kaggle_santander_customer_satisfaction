'''
    This file is python implementation of wpppj's solution in R.

    All credits to the code goes to his original link :
    https://github.com/pjpan/Practice/tree/master/Kaggle-SantanderCustomerSatisfaction
'''

import time

from utils.data_utils import *
from utils.xgb_utils import *
from utils.log_utils import *

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from datetime import datetime

LOG = get_logger('34_wpppj_solution.log')

import numpy as np
import pandas as pd

np.random.seed(777)


def main():

    LOG.info('=' * 50)
    LOG.info('# Santander Customer Satisfaction _ 34_wpppj')
    LOG.info('-' * 50)

    # preprocess data
    prepare_data(LOG)

    ##################################################################################################################
    ### Loading data
    ##################################################################################################################

    LOG.info('=' * 50)
    LOG.info('# Loading data..')
    LOG.info('-' * 50)

    trn = pd.read_csv('./input/trn_x.csv')
    tst = pd.read_csv('./input/tst_x.csv')

    y = trn['TARGET'].values
    tst_id = tst['ID'].values

    drop_cols = ['ID', 'TARGET']
    trn.drop(drop_cols, axis=1, inplace=True)
    tst.drop(drop_cols, axis=1, inplace=True)

    # use xgboost to get 10-fold cv
    vld_stack, tst_stack = xgb_engine(trn, tst, y, LOG)

    # fit stack
    model = LogisticRegression(C=1, n_jobs=-1, random_state=777)
    vld_stack = np.expand_dims(vld_stack, axis=1)
    model.fit(vld_stack, y)
    vld_stack_pred = model.predict_proba(vld_stack)[:,1]
    LOG.info('# Evaluation on vld after LR fit: {:.6}'.format(roc_auc_score(y, vld_stack_pred)))

    # normalize and apply stack
    tst_nrm_stack = (tst_stack - tst_stack.min()) / (tst_stack.max() - tst_stack.min())
    tst_stack_pred = model.predict_proba(np.expand_dims(tst_nrm_stack, axis=1))[:, 1]

    # generate submission
    submission = pd.DataFrame({'ID': tst_id, 'TARGET': tst_stack_pred})
    now = datetime.now()
    if not os.path.exists('./output'):
        os.mkdir('./output')
    sub_file = './output/submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    submission.to_csv(sub_file, index=False)

    # generate submission without LR fit
    submission = pd.DataFrame({'ID': tst_id, 'TARGET': tst_nrm_stack})
    sub_file = './output/submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '_woLR.csv'
    submission.to_csv(sub_file, index=False)


if __name__ == '__main__':
    start = time.time()
    main()
    LOG.info('=' * 50)
    LOG.info('# Finished ({:.2f} sec elapsed)'.format(time.time() - start))
    LOG.info('=' * 50)
