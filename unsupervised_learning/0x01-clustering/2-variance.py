#!/usr/bin/env python3
"""Calculate intracluster variance for a data set"""


import numpy as np


def variance(X, C):
    """
    Calculate intracluster variance for a data set
    X: (n, d) ndarray n = data, d = dimensions
    C: (k, d) ndarray k = cluster centroid, d = dimensions
    """
    if ((not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray) or
         len(X.shape) != 2 or len(C.shape) != 2 or C.shape[1] < 1 or
         X.shape[1] < 1 or X.shape[0] < C.shape[0] or C.shape[0] < 1 or
         X.shape[1] != C.shape[1])):
        return None
    assignments = np.zeros(X.shape[0], dtype=int)
    for cidx, centroid in enumerate(C[1:], 1):
        for pidx, point in enumerate(X):
            assigned = assignments[pidx]
            assigned = pow(point - C[assigned], 2).sum()
            checking = pow(point - centroid, 2).sum()
            if checking < assigned:
                assignments[pidx] = cidx
    variance = 0
    for cidx in range(len(C)):
        assigned = np.where(assignments == cidx, True, False)
        variance += pow(X[assigned, :] - C[cidx], 2).sum()
    return variance
