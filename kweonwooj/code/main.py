'''
    This file is final version of kweonooj.

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
    print('# Santander Customer Satisfaction _ bare_minimum')
    print('-' * 50)

    ##################################################################################################################
    ### Loading data
    ##################################################################################################################

    print('=' * 50)
    print('# Loading data..')
    print('-' * 50)
    X, Y, x_id = load_train()

    ##################################################################################################################
    ### Feature Engineering
    ##################################################################################################################

    print('=' * 50)
    print('# Feature Engineering..')
    print('-' * 50)

    cols = X.columns
    trn = X.as_matrix(columns=cols)

    model = RandomForestClassifier(max_depth=10,
                                   n_jobs=-1, random_state=777)
    model.fit(trn, Y)

    feat_imp = np.zeros(X.shape[1]).tolist()
    for i, j in enumerate(scipy.stats.rankdata(model.feature_importances_, method='ordinal')):
        feat_imp[j - 1] = (cols[i], model.feature_importances_[i])
    feat_imp.reverse()

    ##################################################################################################################
    ### Cross Validations
    ##################################################################################################################

    print('=' * 50)
    print('# Performing Cross-Validation..')
    print('-' * 50)

    param = {
        'objective': 'binary:logistic',
        'eta': 0.1,
        'min_child_weight': 10,
        'max_depth': 8,
        'silent': 1,
        # 'nthread': 16,
        'eval_metric': 'auc',
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.9,
    }
    best_ntree_limits = []

    y_pred = np.zeros(Y.shape)

    n_feat = 70
    cols = [feat_imp[i][0] for i in range(n_feat)]
    trn = X.as_matrix(columns=cols)

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits)
    for i, (trn_index, vld_index) in enumerate(skf.split(trn, Y)):
        print('# CV Iter {} / {}'.format(i+1, n_splits))

        X_trn, X_vld = trn[trn_index], trn[vld_index]
        y_trn, y_vld = Y[trn_index], Y[vld_index]

        dtrn = xgb.DMatrix(X_trn, label=y_trn, feature_names=cols)
        dvld = xgb.DMatrix(X_vld, label=y_vld, feature_names=cols)

        evallist = [(dtrn, 'train'), (dvld, 'eval')]
        model = xgb.train(param, dtrn, 1000, evals=evallist, early_stopping_rounds=20)

        y_pred[vld_index] = model.predict(dvld, ntree_limit=model.best_ntree_limit)
        best_ntree_limits.append(model.best_ntree_limit)


    print('# Evaluate Cross-Validation..')
    cv_score = roc_auc_score(Y, y_pred)
    print('    CV score : {}'.format(cv_score))

    ##################################################################################################################
    ### Prediction
    ##################################################################################################################

    print('# Re-Training on full train data..')
    dtrn = xgb.DMatrix(trn, label=Y, feature_names=cols)
    model = xgb.train(param, dtrn, int(np.mean(best_ntree_limits) * (n_splits + 1) / n_splits))

    print('# Making predictions on test data..')
    X_tst, tst_id = load_test()
    X_tst = X_tst.as_matrix(columns=cols)
    dtst = xgb.DMatrix(X_tst, feature_names=cols)
    tst_pred = model.predict(dtst)

    ##################################################################################################################
    ### Submission
    ##################################################################################################################

    print('# Generating a submission..')
    result = pd.DataFrame(tst_pred, columns=['TARGET'])
    result['ID'] = tst_id

    now = datetime.now()
    if not os.path.exists('./output'):
        os.mkdir('./output')
    sub_file = './output/submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result.to_csv(sub_file, index=False)

    print('=' * 50)

if __name__ == '__main__':
    start = time.time()
    main()
    print('# Finished ({:.2f} sec elapsed)'.format(time.time() - start))

