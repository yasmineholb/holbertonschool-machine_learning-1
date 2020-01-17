#!/usr/bin/env python3
"""Find optimum number of clusters by variance"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Find optimum number of clusters by variance
    X: Data set. (num points, dimensions) ndarray
    kmin: minimum clusters
    kmax: maximum clusters
    iterations: max iterations per variance calculation
    """
    if not isinstance(kmin, int) or kmin < 1:
        raise ValueError("kmin must be a positive integer value")
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax < 1:
        raise ValueError("kmax must be a positive integer value")
    if kmin >= kmax:
        raise ValueError("kmax must be greater than kmin")
    results = []
    delta_variances = [0.0]
    result = kmeans(X, kmin, iterations)
    mink_variance = variance(X, result[0])
    results.append(result)
    for k in range(kmin + 1, kmax + 1):
        result = kmeans(X, k, iterations)
        var = variance(X, result[0])
        results.append(result)
        delta_variances.append(mink_variance - var)
    return results, delta_variances
