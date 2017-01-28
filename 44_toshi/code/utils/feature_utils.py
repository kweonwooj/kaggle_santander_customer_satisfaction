
"""
    This file has a utility function for feature engineering
"""

import numpy as np
import pandas as pd

def generate_name_feature(LOG):

    LOG.info('# Load data')
    trn = pd.read_csv('../input/train.csv')
    tst = pd.read_csv('../input/test.csv')

    add_trn = pd.DataFrame(trn['ID'], columns=['ID'])
    add_tst = pd.DataFrame(tst['ID'], columns=['ID'])

    LOG.info('# Lv.1')
    col = 'num_zero'
    add_trn[col] = (trn == 0).sum(axis=1)
    add_tst[col] = (tst == 0).sum(axis=1)

    col = 'num_zero_delta'
    cols = [c for c in trn.columns if 'delta' in c]
    add_trn[col] = (trn[cols] == 0).sum(axis=1)
    add_tst[col] = (tst[cols] == 0).sum(axis=1)

    col = 'num_zero_imp'
    cols = [c for c in trn.columns if 'imp' in c]
    add_trn[col] = (trn[cols] == 0).sum(axis=1)
    add_tst[col] = (tst[cols] == 0).sum(axis=1)

    col = 'num_zero_ind'
    cols = [c for c in trn.columns if 'ind' in c]
    add_trn[col] = (trn[cols] == 0).sum(axis=1)
    add_tst[col] = (tst[cols] == 0).sum(axis=1)

    col = 'num_zero_num'
    cols = [c for c in trn.columns if 'num' in c]
    add_trn[col] = (trn[cols] == 0).sum(axis=1)
    add_tst[col] = (tst[cols] == 0).sum(axis=1)

    col = 'num_zero_saldo'
    cols = [c for c in trn.columns if 'saldo' in c]
    add_trn[col] = (trn[cols] == 0).sum(axis=1)
    add_tst[col] = (tst[cols] == 0).sum(axis=1)

    col = 'num_zero_var'
    cols = [c for c in trn.columns if 'var' in c]
    add_trn[col] = (trn[cols] == 0).sum(axis=1)
    add_tst[col] = (tst[cols] == 0).sum(axis=1)

    col = 'num_na'
    add_trn[col] = (trn.isnull()).sum(axis=1)
    add_tst[col] = (tst.isnull()).sum(axis=1)


    LOG.info('# Lv.2')
    columns = ['num_zero_delta_imp', 'num_zero_delta_num', 'num_zero_imp_aport', 'num_zero_imp_op', 'num_zero_num_meses',
     'num_zero_num_op', 'num_zero_num_trasp', 'num_zero_num_var', 'num_zero_saldo_medio', 'num_zero_saldo_var']

    for col in columns:
        cols = [c for c in trn.columns if col[9:] in c]
        add_trn[col] = (trn[cols] == 0).sum(axis=1)
        add_tst[col] = (tst[cols] == 0).sum(axis=1)

    LOG.info('# Save files')
    add_trn.to_csv('./input/name_feat_trn.csv', index=False)
    add_tst.to_csv('./input/name_feat_tst.csv', index=False)
