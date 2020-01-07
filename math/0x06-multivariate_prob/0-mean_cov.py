#!/usr/bin/env python3
"""Calculate mean and covariance of a data set"""


import numpy as np


def mean_cov(X):
    """Calculate mean and covariance of a data set"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    means = X.mean(axis=0)
    covmat = np.ndarray((X.shape[1], X.shape[1]))
    print(X.shape)
    for row in range(X.shape[1]):
        for col in range(X.shape[1]):
            if row > col:
                covmat[row][col] = covmat[col][row]
                continue
            if row == col:
                covmat[row][col] = pow(X[:, row] - means[row], 2).mean()
                continue
            covmat[row][col] = np.multiply(X[:, row] - means[row],
                                           X[:, col] - means[col]).mean()
    return means[np.newaxis, :], covmat
