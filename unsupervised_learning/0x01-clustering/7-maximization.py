#!/usr/bin/env python3
"""Do an expectation maximization step"""


import numpy as np


def maximization(X, g):
    """
    Do an expectation maximization step
    X: data. (num points, dims)
    g: posterior probabilities. (clusters, num points)
    """
    if ((not isinstance(X, np.ndarray) or not isinstance(g, np.ndarray) or
         len(X.shape) != 2 or len(g.shape) != 2 or X.shape[0] != g.shape[1] or
         X.shape[0] < 1 or X.shape[1] < 1 or g.shape[0] < 1 or
         g.shape[0] > g.shape[1])):
        return None, None, None
    for total in g.sum(axis=0):
        if not np.isclose(total, 1):
            return None, None, None
    try:
        pi = np.empty(g.shape[0])
        for cidx in range(pi.shape[0]):
            pi[cidx] = (g[cidx] / X.shape[0]).sum()
            means = np.empty((g.shape[0], X.shape[1]))
            for cidx in range(g.shape[0]):
                means[cidx] = (g[cidx, None].T * X).sum(axis=0) / g[cidx].sum()
            covs = np.empty((g.shape[0], X.shape[1], X.shape[1]))
            for cidx in range(g.shape[0]):
                for dm1 in range(X.shape[1]):
                    for dm2 in range(X.shape[1]):
                        a = X[:, dm1] - means[cidx, dm1]
                        b = X[:, dm2] - means[cidx, dm2]
                        c = a * b
                        covs[cidx, dm1, dm2] = (c
                                                / X.shape[0] / pi[cidx]
                                                * g[cidx]).sum()
        return pi, means, covs
    except Exception:
        return None, None, None
