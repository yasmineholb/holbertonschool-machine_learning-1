#!/usr/bin/env python3
"""Calculate weights matrix that maintains at least some given variance"""


import numpy as np


def pca(X, ndim):
    """Calculate weights matrix that maintains at least some given variance"""
    xmean = X - np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(xmean)
    return np.dot(xmean, vh.T[:, :ndim])
    return np.dot(u[:ndim], np.diag(s[:ndim]))
