#!/usr/bin/env python3
"""Calculate steady state of a regular markov chain"""


import numpy as np


def absorbing(P):
    """
    Calculate steady state of a regular markov chain
    P: the transition matrix, (n, n) ndarray
    """
    if ((not isinstance(P, np.ndarray) or len(P.shape) != 2 or P.shape[0] < 1
         or P.shape[0] != P.shape[1] or np.where(P < 0, 1, 0).any()
         or not np.where(np.isclose(P.sum(axis=1), 1), 1, 0).any())):
        return None
    absorbers = np.zeros(P.shape[0])
    prev = absorbers.copy()
    for i in range(P.shape[0]):
        if P[i][i] == 1:
            absorbers[i] = 1
    while (not np.array_equal(absorbers, prev)
           and absorbers.sum() != P.shape[0]):
        prev = absorbers.copy()
        for absorbed in P[:, np.nonzero(absorbers)[0]].T:
            absorbers[np.nonzero(absorbed)] = 1
    if absorbers.sum() == P.shape[0]:
        return True
    return False
