#!/usr/bin/env python3
"""Calculate PDF of a multivariate gaussian distribution"""


import numpy as np


def pdf(X, m, S):
    """
    Calculate PDF of a multivariate gaussian distribution
    X: data. (num points, dims)
    m: means. (dims)
    S: covariance matrix. (dims, dims)
    """
    if ((not isinstance(X, np.ndarray) or not isinstance(m, np.ndarray) or
         not isinstance(S, np.ndarray) or len(X.shape) != 2 or
         len(m.shape) != 1 or len(S.shape) != 2 or X.shape[0] < 1 or
         X.shape[1] < 1 or
         not (X.shape[1] == m.shape[0] == S.shape[1] == S.shape[0]))):
        return None
    pdfs = np.empty(X.shape[0])
    for idx, x in enumerate(X):
        pdfs[idx] = (np.exp(np.dot(np.dot((x - m), np.linalg.inv(S)),
                                   (x - m).T) / -2) /
                     np.sqrt(pow(2 * np.pi, X.shape[1]) * np.linalg.det(S)))
    pdfs = np.where(pdfs > 1e-300, pdfs, 1e-300)
    return pdfs
