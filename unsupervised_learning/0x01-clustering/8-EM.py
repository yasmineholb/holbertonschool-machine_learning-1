#!/usr/bin/env python3
"""Run expectation maximization"""


import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Run expectation maximization
    X: input data. (num points, dims)
    k: number of clusters
    iterations: max number of iterations
    tol: delta change to stop at
    """
    try:
        if iterations < 1 or not isinstance(verbose, bool) or tol < 0:
            return None, None, None, None, None
    except Exception:
        return None, None, None, None, None

    pi, means, covs = initialize(X, k)
    if pi is None or means is None or covs is None:
        return None, None, None, None, None
    itrs = 0
    prevlikely = 1
    for itrs in range(iterations):
        expects, loglikely = expectation(X, pi, means, covs)
        if expects is None or loglikely is None:
            return None, None, None, None, None
        pi, means, covs = maximization(X, expects)
        if pi is None or means is None or covs is None:
            return None, None, None, None, None
        if verbose and not itrs % 10:
            print("Log likelihood after {} iterations: {}"
                  .format(itrs, loglikely))
        if abs(prevlikely - loglikely) < tol:
            break
        prevlikely = loglikely
    if verbose and itrs % 10:
        print("Log likelihood after {} iterations: {}".format(itrs, loglikely))
    return pi, means, covs, expects, loglikely
