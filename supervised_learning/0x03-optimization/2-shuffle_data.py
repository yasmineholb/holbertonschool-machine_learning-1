#!/usr/bin/env python3
"""Shuffle data in two matrices in the same way"""


import numpy as np


def shuffle_data(X, Y):
    """Shuffle data in two matrices in the same way"""
    shufflidx = np.random.permutation(X.shape[0])
    return X[shufflidx], Y[shufflidx]
