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
        covmat = np.ndarray((X.shape[0], X.shape[0]))
        for row in range(X.shape[0]):
            for col in range(X.shape[0]):
                if row > col:
                    covmat[row][col] = covmat[col][row]
                    continue
                covmat[row][col] = (np.multiply(X[row] - means[row],
                                                X[col] - means[col]).sum()
                                    / (X.shape[1] - 1))
        return means[:, np.newaxis], covmat

    def pdf(self, x):
        """Calculate PDF at point x"""

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if ((len(x.shape) != 2 or x.shape[1] != 1
             or x.shape[0] != self.mean.shape[0])):
            raise ValueError("x must have the shape ({}, 1)"
                             .format(self.mean.shape[0]))
        centered = x - self.mean
        return (np.exp(np.dot(np.dot(centered.T,
                                     np.linalg.inv(self.cov)),
                              centered) / -2) /
                np.sqrt(pow(2 * np.pi, self.mean.shape[0])
                        * np.linalg.det(self.cov)))[0, 0]
