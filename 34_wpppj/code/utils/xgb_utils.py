"""
    This file has a utility function for model.
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

np.random.seed(777)

def xgb_engine(trn, tst, y, LOG):

    features = trn.columns
    trn = trn.as_matrix(columns=features)
    tst = tst.as_matrix(columns=features)

    n_folds = 10
    skf = StratifiedKFold(n_splits=n_folds)

    param = {
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'eval_metric': 'auc',
        'eta': 0.02,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 1
    }

    vld_stack = np.zeros(trn.shape[0])
    tst_stack = np.zeros(tst.shape[0])
    dtst = xgb.DMatrix(tst, feature_names=features)

    for i, (trn_ind, vld_ind) in enumerate(skf.split(trn, y)):

        LOG.info('# Fold {} / {}'.format(i + 1, n_folds))

        x_trn, x_vld = trn[trn_ind], trn[vld_ind]
        y_trn, y_vld = y[trn_ind], y[vld_ind]

        dtrn = xgb.DMatrix(x_trn, label=y_trn, feature_names=features)
        dvld = xgb.DMatrix(x_vld, label=y_vld, feature_names=features)

        if i == 0:
            n_rounds = 1200
            best_cv = xgb.cv(params=param,
                             dtrain=dtrn,
                             num_boost_round=n_rounds,
                             nfold=n_folds,
                             early_stopping_rounds=20,
                             maximize=True,
                             verbose_eval=100,
                             seed=777)
            LOG.info('# Best n_tree : {}'.format(best_cv['test-auc-mean'].max()))

        n_rounds = best_cv['test-auc-mean'].idxmax()
        xgb_stack = xgb.train(params=param,
                              dtrain=dtrn,
                              num_boost_round=n_rounds,
                              verbose_eval=int(n_rounds / 10),
                              maximize=True)

        vld_stack[vld_ind] = xgb_stack.predict(dvld)
        tst_stack += xgb_stack.predict(dtst) * 1.0 / n_folds

    LOG.info('# Evaluation on vld : {:.6}'.format(roc_auc_score(y, vld_stack)))

    return vld_stack, tst_stack
