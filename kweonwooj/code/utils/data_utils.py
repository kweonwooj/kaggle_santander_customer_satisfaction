
"""
    This file has a utility function for data IO
"""

import pandas as pd
import numpy as np
import itertools

def load_train():
    train_path = '../input/train.csv'
    data = pd.read_csv(train_path)

    y_col = 'TARGET'
    Y = data[y_col].values

    id_col = 'ID'
    x_id = data[id_col].values

    X = data.drop([y_col, id_col], axis=1)

    return X, Y, x_id

def load_test():
    test_path = '../input/test.csv'
    data = pd.read_csv(test_path)

    id_col = 'ID'
    x_id = data[id_col].values

    X = data.drop([id_col], axis=1)

    return X, x_id
