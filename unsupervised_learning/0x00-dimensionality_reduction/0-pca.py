#!/usr/bin/env python3
"""Calculate weights matrix that maintains at least some given variance"""


import numpy as np


def pca(X, var=.95):
    """Calculate weights matrix that maintains at least some given variance"""
    u, s, vh = np.linalg.svd(X)
    sumvar = s[0]
    end = 0
    totalvar = s.sum() * var
    while (sumvar < totalvar):
        end += 1
        sumvar += s[end]
    return vh.T[:, :end + 1]
