#!/usr/bin/env python3
"""Calculate expectation maximization step for a Gaussian Mixture Model"""


import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculate expectation maximization step for a Gaussian Mixture Model
    X: data. (points, dims) ndarray
    pi: priors for each cluster. (cluster) ndarray
    m: centroid means. (clusters, dims) ndarray
    S: covariance matrices for clusters. (clusters, dims, dims) ndarray
    """
    try:
        if ((not isinstance(X, np.ndarray) or not isinstance(pi, np.ndarray) or
             not isinstance(m, np.ndarray) or not isinstance(S, np.ndarray) or
             X.shape[0] < 1 or pi.shape[0] < 1 or pi.shape[0] > X.shape[0] or
             len(X.shape) != 2 or len(pi.shape) != 1 or len(m.shape) != 2 or
             len(S.shape) != 3 or
             X.shape[1] != m.shape[1] or S.shape[1] != S.shape[2] or
             X.shape[1] != S.shape[2] or pi.shape[0] != m.shape[0] or
             not np.isclose(pi.sum(), 1))):
            return None, None
        pdfs = np.empty((pi.shape[0], X.shape[0]))
        for cidx in range(pi.shape[0]):
            pdfs[cidx] = pi[cidx] * pdf(X, m[cidx], S[cidx])
            expects = np.empty((pi.shape[0], X.shape[0]))
            for cidx in range(expects.shape[0]):
                for pidx in range(expects.shape[1]):
                    pipdf = pi[cidx] * pdfs[cidx, pidx]
                    expects[cidx][pidx] = (pdfs[cidx, pidx] /
                                           (pdfs[:, pidx]).sum())
        return expects, np.log(pdfs.sum(axis=0)).sum()
    except Exception:
        return None, None
