import numpy as np
import pandas as pd
np.random.seed(12324)
from santander_models import *
import sys, getopt, re
import os
from santander_preprocess import *
from sklearn.metrics import roc_auc_score

def main_20sets(LOG):
    LOG.info('*'*50)
    LOG.info('# main_20sets..')

    # input & output path redefined
    INPUT_PATH = '../input/'
    OUTPUT_PATH = './output/'

    MODELS_ALL = ['xgboost']
    FEATURES_ALL = [['SumZeros', 'pca', 'likeli']]

    train = pd.read_csv(INPUT_PATH + 'train.csv')
    test = pd.read_csv(INPUT_PATH + 'test.csv')

    # generate 5fold_20times cv indices
    n_split = 5
    for i in range(1, 21):
        if i == 1:
            cv_split = pd.DataFrame(
                {'set' + str(i): (np.random.permutation(train.shape[0]) / (train.shape[0] / n_split)).astype(int) + 1})
        else:
            cv_split['set' + str(i)] = (np.random.permutation(train.shape[0]) / (train.shape[0] / n_split)).astype(int) + 1
    cv_split.to_csv('../input/5fold_20times.csv', index=False)

    preds_all = train[['ID']].append(test[['ID']], ignore_index=True).copy()
    # Iterate over model choices defined at MODELS_ALL
    for imod in range(len(MODELS_ALL)):
        MODEL = MODELS_ALL[imod]
        FEATURES = FEATURES_ALL[imod]
        LOG.info('# Training {}...'.format(MODEL))

        train = pd.read_csv(INPUT_PATH + 'train.csv')
        test = pd.read_csv(INPUT_PATH + 'test.csv')
        id_fold = pd.read_csv(INPUT_PATH + '5fold_20times.csv')
        id_fold['ID'] = train['ID'].values

        # Do unique feature engineering
        train, test = process_base(train, test)
        train, test = drop_sparse(train, test)
        train, test = drop_duplicated(train, test)
        train, test = add_features(train, test, FEATURES)

        flist = [x for x in train.columns if not x in ['ID', 'TARGET']]

        preds_model = pd.DataFrame()
        # Initiate CV for multiple iterations
        for it in range(1, 21):
            LOG.info('# Processing iteration {}...'.format(it))
            # extract cv index from each set
            it_id_fold = id_fold[['ID', 'set' + str(it)]]
            it_id_fold.columns = ['ID', 'FOLD']
            if 'FOLD' in train.columns:
                train.drop('FOLD', axis=1, inplace=True)
            train = pd.merge(train, it_id_fold, on='ID', how='left')
            aucs = []

            # 5-fold Cross-validation
            for fold in range(5):
                train_split = train[train['FOLD'] != fold + 1].copy().reset_index(drop=True)
                val_split = train[train['FOLD'] == fold + 1].copy().reset_index(drop=True)
                test_split = val_split[['ID'] + flist].append(test[['ID'] + flist], ignore_index=True)
                ids_val = val_split['ID'].values

                if 'likeli' in FEATURES:
                    train_split, test_split, flist1 = add_likelihood_feature('saldo_var13', train_split, test_split,
                                                                             flist)
                else:
                    flist1 = flist

                X_train = train_split[flist1].values
                y_train = train_split['TARGET'].values
                X_test = test_split[flist1].values

                # Train and Predict on models
                if MODEL == 'xgboost':
                    y_pred = train_predict_xgboost_bugged(X_train, y_train, X_test)

                if MODEL == 'adaboost_classifier':
                    y_pred = train_predict_adaboost_classifier(X_train, y_train, X_test)

                # Save prediction with fold/iter info
                preds = pd.DataFrame()
                preds['ID'] = test_split['ID'].values
                preds['FOLD'] = fold
                preds['ITER'] = it
                preds[MODEL] = y_pred

                # contains both valid and test predictions
                preds_model = preds_model.append(preds, ignore_index=True)

                preds = preds.loc[preds['ID'].isin(ids_val)].copy()
                preds = pd.merge(preds, train[['ID', 'TARGET']], on='ID', how='left')

                # AUC calculation
                fold_auc = roc_auc_score(preds['TARGET'], preds[MODEL])
                aucs.append(fold_auc)
            LOG.info(np.mean(aucs), np.std(aucs))

        # clip prediction values
        preds_model.loc[preds_model[MODEL] < 0, MODEL] = 0.0
        preds_model.loc[preds_model[MODEL] > 1, MODEL] = 1.0

        # Obtain average of 20 sets
        preds_model = preds_model.groupby(['ID', 'ITER'])[MODEL].mean().reset_index()
        # Get rank
        for it in range(1, 21):
            preds_model.loc[preds_model['ITER'] == it, MODEL] = preds_model.loc[preds_model['ITER'] == it, MODEL].rank()
        preds_model = preds_model.groupby('ID')[MODEL].mean().reset_index()
        preds_model.columns = ['ID', 'dmitry_' + MODEL]
        preds_all = pd.merge(preds_all, preds_model, on='ID', how='left')
        preds_all.to_csv('all_models_temp.csv', index=False)

    # Extract train cv predictions
    preds_train = pd.merge(train[['ID']], preds_all, on='ID', how='left')
    preds_train.to_csv(OUTPUT_PATH + 'dmitry_train.csv', index=False)
    # Extract test predictions
    preds_test = pd.merge(test[['ID']], preds_all, on='ID', how='left')
    preds_test.to_csv(OUTPUT_PATH + 'dmitry_test.csv', index=False)
    LOG.info("# Done training!")