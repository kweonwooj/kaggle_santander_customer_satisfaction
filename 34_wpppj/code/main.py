'''
    This file is python implementation of wpppj's solution in R.

    All credits to the code goes to his original link :
    https://github.com/pjpan/Practice/tree/master/Kaggle-SantanderCustomerSatisfaction
'''

import time

from utils.data_utils import *
from utils.xgb_utils import *
from utils.log_utils import *

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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

    features = trn.columns
    trn = trn.as_matrix(columns=features)
    tst = tst.as_matrix(columns=features)

    # classifiers to use in blending
    clfs = [RandomForestClassifier(n_estimators=400, n_jobs=-1, criterion='gini', random_state=777),
            ExtraTreesClassifier(n_estimators=400, n_jobs=-1, criterion='gini', random_state=777),
            GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=400,
                                       random_state=777),
            LogisticRegression(C=1, n_jobs=-1, random_state=777),
            xgb.XGBClassifier(missing=np.nan, max_depth=5, n_estimators=500, learning_rate=0.02, subsample=0.7,
                              colsample_bytree=0.7)
            ]

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds)

    blend_trn = np.zeros((trn.shape[0], len(clfs)))
    blend_tst = np.zeros((tst.shape[0], len(clfs)))

    # blender
    for j, clf in enumerate(clfs):
        LOG.info('# {} / {} : {}'.format(j + 1, len(clfs), clf))
        blend_tst_j = np.zeros(tst.shape[0], n_folds)

        for i, (trn_ind, vld_ind) in enumerate(skf.split(trn, y)):
            LOG.info('# Fold {} / {}'.format(i + 1, n_folds))

            x_trn, x_vld = trn[trn_ind], trn[vld_ind]
            y_trn, y_vld = y[trn_ind], y[vld_ind]

            clf.fit(x_trn, y_trn)

            vld_pred = clf.predict_proba(x_vld)[:, 1]
            blend_trn[vld_ind, j] = vld_pred

            tst_pred = clf.predict_proba(tst)[:, 1]
            blend_tst_j[:, i] = tst_pred
        blend_tst[:, j] = blend_tst_j.mean(1)

    # blend using Logistic Regression
    LOG.info('# Blending..')
    clf = LogisticRegression(C=1, n_jobs=-1, random_state=777)

    # blend cv test by 10-fold
    vld_pred = np.zeros((trn.shape[0], 1))
    for i, (trn_ind, vld_ind) in enumerate(skf.split(trn, y)):
        LOG.info('# Fold {} / {}'.format(i + 1, n_folds))

        x_trn, x_vld = trn[trn_ind], trn[vld_ind]
        y_trn, y_vld = y[trn_ind], y[vld_ind]

        clf.fit(x_trn, y_trn)
        vld_pred[vld_ind] = clf.predict_proba(x_vld)[:,1]

    cv_score = roc_auc_score(y, vld_pred)
    LOG.info('# CV Score: {}'.format(cv_score))

    # refit all data
    clf.fit(blend_trn, y)
    tst_pred = clf.predict_proba(blend_tst)[:, 1]
    tst_pred = (tst_pred - tst_pred.min()) / (tst_pred.max() - tst_pred.min())

    # make submission file
    submission = pd.DataFrame({'ID': tst_id, 'TARGET': tst_pred})
    now = datetime.now()
    if not os.path.exists('./output'):
        os.mkdir('./output')
    sub_file = './output/submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    submission.to_csv(sub_file, index=False)


if __name__ == '__main__':
    start = time.time()
    main()
    LOG.info('=' * 50)
    LOG.info('# Finished ({:.2f} sec elapsed)'.format(time.time() - start))
    LOG.info('=' * 50)
