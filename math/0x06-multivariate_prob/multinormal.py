#!/usr/bin/env python3
"""Multivariate normal distribution stats class"""


import numpy as np


class MultiNormal:
    """Multivariate normal distribution stats class"""
    def __init__(self, data):
        """
        data: the data points to build the stats from
        """
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        self.mean, self.cov = self.mean_cov(data)

    def mean_cov(self, X):
        """Calculate mean and covariance of a data set"""
        means = X.mean(axis=1)
        covmat = np.ndarray((3, 3))
        for row in range(X.shape[0]):
            for col in range(X.shape[0]):
                if row > col:
                    covmat[row][col] = covmat[col][row]
                    continue
                if row == col:
                    covmat[row][col] = pow(X[row] - means[row], 2).mean()
                    continue
                covmat[row][col] = np.multiply(X[row] - means[row],
                                               X[col] - means[col]).mean()
        return means, covmat

    def pdf(self, x):
        """Calculate PDF at point x"""
        return None
