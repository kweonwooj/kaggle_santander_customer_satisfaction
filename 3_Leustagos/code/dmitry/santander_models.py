import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import os
import glob


def train_predict_adaboost_classifier(X_train, y_train, X_test):
    clf = AdaBoostClassifier(n_estimators=300, learning_rate=0.1, random_state=32934)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:,1]
    return y_pred

def train_predict_xgboost_bugged(X_train, y_train, X_test):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.02
    param['max_depth'] = 5
    param['eval_metric'] = 'auc'
    param['silent'] = 1
    #param['nthread'] = 10 # set it to max
    param['gamma'] = 1.0
    param['min_child_weight'] = 5
    param['subsample'] = 0.8
    param['colsample_bytree'] = 1.0
    param['colsample_bylevel'] = 0.7
    num_round = 500

    y_pred = [0.0]*len(X_test)
    seeds = [123089, 21324, 324003, 450453, 120032]
    for seed in seeds:
        param['seed'] = seed
        plst = list(param.items())
        xgmat_train = xgb.DMatrix(X_train, label=y_train, missing = -999.0)
        xgmat_test = xgb.DMatrix(X_test, missing = -999.0)
        bst = xgb.train(plst, xgmat_train, num_round)
        y_pred = y_pred + bst.predict( xgmat_test )
    y_pred = y_pred * 1.0 / len(seeds)
    return y_pred
