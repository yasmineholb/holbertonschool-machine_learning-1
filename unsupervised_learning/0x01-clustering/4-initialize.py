#!/usr/bin/env python3
"""Initialize a Gaussian Mixture Model"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initialize a Gaussian Mixture Model
    X: data points. (nun points, dimensions) ndarray
    k: number of clusters
    """
    centroids = kmeans(X, k)[0]
    if centroids is None:
        return None, None, None
    idents = np.empty((k, X.shape[1], X.shape[1]))
    for idx in range(len(idents)):
        idents[idx] = np.identity(X.shape[1])
    return (np.full(k, 1 / k, dtype=float), centroids, idents)
