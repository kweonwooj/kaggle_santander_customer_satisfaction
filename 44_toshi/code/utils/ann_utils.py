"""
    This file has a utility function for ann_model.
"""

import numpy as np
import pandas as pd
from glob import glob
import os
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from keras.models import Sequential
from keras.layers import Dense, Dropout, LeakyReLU
from keras.optimizers import SGD

np.random.seed(777)


def ann_model(hidden_1, hidden_2, relu_1, relu_2, dropout_1, dropout_2, input_dim):
    model = Sequential()

    model.add(Dense(hidden_1, input_dim=input_dim))
    model.add(LeakyReLU(relu_1))
    model.add(Dropout(dropout_1))

    model.add(Dense(hidden_2))
    model.add(LeakyReLU(relu_2))
    model.add(Dropout(dropout_2))

    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.001, momentum=0.5, decay=1e-7, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model

def ann_engine(trn, tst, y, test_id, LOG):

    # normalize
    nn = Normalizer()
    trn = nn.fit_transform(trn)
    tst = nn.transform(tst)

    iters = 30
    for i in range(iters):
        LOG.info('# Iter {} / {}'.format(i+1, iters))

        hidden_1 = np.random.choice(256 - 64, 1)[0] + 64
        hidden_2 = np.random.choice(256 - 32, 1)[0] + 64
        relu_1 = int(np.random.choice(3, 1)[0] + 1) / 10
        relu_2 = int(np.random.choice(3, 1)[0] + 1) / 10
        dropout_1 = int(np.random.choice(6, 1)[0] + 1) / 10
        dropout_2 = int(np.random.choice(6, 1)[0] + 1) / 10


        # cross validation to obtain best_itr
        y_pred = np.zeros(y.shape[0])
        n_splits = 5
        nb_epoch = 30
        skf = StratifiedKFold(n_splits=n_splits, random_state=777)
        for j, (trn_index, vld_index) in enumerate(skf.split(trn, y)):
            print('# CV Iter {} / {}'.format(j + 1, n_splits))

            x_trn, x_vld = trn[trn_index], trn[vld_index]
            y_trn, y_vld = y[trn_index], y[vld_index]

            model = ann_model(hidden_1, hidden_2, relu_1, relu_2, dropout_1, dropout_2, trn.shape[1])
            model.fit(x_trn, y_trn, nb_epoch=nb_epoch)

            y_pred[vld_index] = model.predict(x_vld)

        cv_score = roc_auc_score(y, y_pred)
        print('# CV Score : {}'.format(cv_score))


        if not os.path.exists('./output'):
            os.mkdir('./output')

        model = ann_model(hidden_1, hidden_2, relu_1, relu_2, dropout_1, dropout_2, trn.shape[1])
        model.fit(x_trn, y_trn, nb_epoch=int(nb_epoch * 1.2))

        submission = pd.DataFrame(model.predict(tst), columns=['TARGET'])
        submission['ID'] = test_id

        if not os.path.exists('./output/ann'):
            os.mkdir('./output/ann')

        filename = './output/ann/ann_itr_{}_valid_{}.csv'.format(i, round(cv_score, 4))
        submission.to_csv(filename, index=False)

def ann_ensemble():
    engine_name = 'ann'
    candidates = glob('./output/{}/*'.format(engine_name))
    scores = [np.float(candi.split('_')[-1].split('.csv')[0]) for candi in candidates]
    info = pd.DataFrame(candidates, columns=['filename'])
    info['TARGET'] = scores

    num_model = 10
    for i in range(num_model):
        target_file = info.sort(columns='TARGET', ascending=False).iloc[i]['filename']
        data_i = pd.read_csv(target_file)

        if i == 0:
            data = data_i
        else:
            data['TARGET'] += data_i['TARGET']

    data['TARGET'] /= num_model
    mean_valid = np.mean(info.sort(columns='TARGET', ascending=False).iloc[:num_model]['TARGET'])
    data.to_csv('./output/{}_nummodel-{}_mvalid-{}.csv'.format(engine_name, num_model, round(mean_valid, 4)),
                index=False)
