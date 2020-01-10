#!/usr/bin/env python3
"""Calculate likelihood"""


import numpy as np


def likelihood(x, n, P):
    """Calculate likelihood"""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal "
                         "to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for prob in P:
        if prob > 1 or prob < 0:
            raise ValueError("All values in P must be in the range [0, 1]")
    npfact = np.math.factorial
    return (npfact(n) / (npfact(x) * npfact(n - x))
            * pow(P, x) * pow((1 - P), n - x))


def intersection(x, n, P, Pr):
    """Calculate intersection"""
    likelihoods = likelihood(x, n, P)
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for prob in Pr:
        if prob > 1 or prob < 0:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(Pr.sum(), 1):
        raise ValueError("Pr must sum to 1")
    return likelihoods * Pr


def marginal(x, n, P, Pr):
    "Calculate marginal probability"
    return intersection(x, n, P, Pr).sum()
