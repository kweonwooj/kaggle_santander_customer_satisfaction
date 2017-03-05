'''
    This file is python implementation of toshi_k's solution in Lua, R.

    Original Link: https://github.com/toshi-k/kaggle-santander-customer-satisfaction
'''

import os
import time
from utils.log_utils import *
from dmitry.tsne_features import *
from dmitry.kmeans_features import *
from dmitry.pca_features import *
from dmitry.main_20sets import *

LOG = get_logger('3_Luestagos.log')
import numpy as np
np.random.seed(777)

# global parameters
run_Dmitry = True

def main():

    LOG.info('=' * 50)
    LOG.info('# Santander Customer Satisfaction _ 3_Luestagos')
    LOG.info('-' * 50)

    if run_Dmitry:
        LOG.info('# Dmitry')
        tsne_features(LOG)
        kmeans_features(LOG)
        pca_features(LOG)
        main_20sets(LOG)


if __name__ == '__main__':
    start = time.time()
    main()
    LOG.info('=' * 50)
    LOG.info('# Finished ({:.2f} sec elapsed)'.format(time.time() - start))
    LOG.info('=' * 50)
