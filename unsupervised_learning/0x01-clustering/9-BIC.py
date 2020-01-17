#!/usr/bin/env python3
"""Calculate BIC for ndimensional data"""


import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Calculate BIC for ndimensional data
    X: input data. (num points, dims)
    kmin: minimum clusters
    kmax: maximum clusters
    iterations: max iterations during fitting
    tol: minimum loglikelihood delta before stopping
    """
    if not isinstance(kmin, int) or kmin < 1:
        raise ValueError("kmin must be a positive integer value")
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax < 1:
        raise ValueError("kmax must be a positive integer value")
    if kmin >= kmax:
        raise ValueError("kmax must be greater than kmin")
    best_k = kmin
    loglikelihoods = np.empty(kmax - kmin + 1)
    bics = np.empty(kmax - kmin + 1)
    result = expectation_maximization(X, kmin, iterations, tol, verbose)
    if result is None or result[0] is None:
        return None
    loglikelihoods[0] = result[4]
    bics[0] = ((1 + 2 * kmin * pow(X.shape[1], 2)) * np.log(X.shape[0]) -
               2 * result[4])
    best_result = result[0:3]
    for k in range(kmin + 1, kmax + 1):
        result = expectation_maximization(X, k, iterations, tol, verbose)
        loglikelihoods[k - kmin] = result[4]
        bics[k - kmin] = ((1 + 2 * k * pow(X.shape[1], 2))
                          * np.log(X.shape[0]) -
                          2 * result[4])
        if bics[k - kmin] < bics[best_k - kmin]:
            best_k = k
            best_result = result[0:3]
    return best_k, best_result, loglikelihoods, bics
