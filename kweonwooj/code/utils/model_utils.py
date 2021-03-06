"""
    This file has a utility function for model.
"""

import xgboost as xgb
import numpy as np
import pandas as pd
import os

np.random.seed(777)
iters = 10
result_all = pd.DataFrame()

def xgb_engine(trn, tst, y, test_id):

    for i in range(iters):
        print('# Iter {} / {}'.format(i+1, iters))

        myparam_colsample_bytree = np.random.uniform(0.5, 0.99, 1)[0]
        myparam_subsample = np.random.uniform(0.7, 0.99, 1)[0]
        myparam_eta = np.random.uniform(0.005, 0.05, 1)[0]
        myparam_max_depth = np.random.choice(5, 1)[0] + 4
        myparam_max_delta_step = np.random.choice(4, 1)[0]
        myparam_base_score = np.random.uniform(0.05, 0.5, 1)[0]

        param = {'objective': 'binary:logistic',
                 'booster': 'gbtree',
                 'eval_metric': 'auc',
                 'colsample_bytree': myparam_colsample_bytree,
                 'subsample': myparam_subsample,
                 'eta': myparam_eta,
                 'max_depth': myparam_max_depth,
                 'max_delta_step': myparam_max_delta_step,
                 'base_score': myparam_base_score
                 }

        features = trn.columns
        dtrn = xgb.DMatrix(trn.as_matrix(columns=features), label=y, feature_names=features)
        model_cv = xgb.cv(params=param,
                          dtrain=dtrn,
                          num_boost_round=100000,
                          nfold=10,
                          stratified=True,
                          metrics='auc',
                          early_stopping_rounds=100,
                          verbose_eval=100,
                          seed=777+i)

        best_score = max(model_cv['test-auc-mean'])
        best_itr = np.argmax(model_cv['test-auc-mean'])

        result_new = pd.DataFrame([{'ID': i,
                                    'myparam_colsample_bytree': myparam_colsample_bytree,
                                    'myparam_subsample': myparam_subsample,
                                    'myparam_eta': myparam_eta,
                                    'myparam_max_depth': myparam_max_depth,
                                    'myparam_max_delta_step': myparam_max_delta_step,
                                    'myparam_base_score': myparam_base_score,
                                    'best_itr': int(best_itr * 1.1),
                                    'best_score': best_score}])
        result_all = pd.concat([result_all, result_new], axis=0)

        if not os.path.exists('./output'):
            os.mkdir('./output')

        result_all.to_csv('./output/result_all.csv', index=False)

        model = xgb.train(dtrain=dtrn, params=param, num_boost_round=best_itr)
        imp = model.get_fscore()

        dtst = xgb.DMatrix(tst.as_matrix(columns=features))
        submission = pd.DataFrame(model.predict(dtst), columns=['TARGET'])
        submission['ID'] = test_id

        if not os.path.exists('./output/xgb'):
            os.mkdir('./output/xgb')

        filename = './output/xgb/xgb_itr_{}_valid_{}.csv'.format(i, round(best_score, 4))
        submission.to_csv(filename, index=False)
