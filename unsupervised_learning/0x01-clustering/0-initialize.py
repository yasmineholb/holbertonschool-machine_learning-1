#!/usr/bin/env python3
"""Initialize k clusters with mutivariate uniform distribution"""


import numpy as np


def initialize(X, k):
    """Initialize k clusters with mutivariate uniform distribution"""
    if ((not isinstance(X, np.ndarray) or len(X.shape) != 2 or
         not isinstance(k, int) or k < 1 or k > X.shape[0] or X.shape[1] < 1)):
        return None
    return np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0),
                             (k, X.shape[1]))
