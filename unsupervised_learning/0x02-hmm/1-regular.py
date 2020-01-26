#!/usr/bin/env python3
"""Calculate steady state of a regular markov chain"""


import numpy as np


def regular(P):
    """
    Calculate steady state of a regular markov chain
    P: the transition matrix, (n, n) ndarray
    """
    if ((not isinstance(P, np.ndarray) or len(P.shape) != 2 or P.shape[0] < 1
         or P.shape[0] != P.shape[1] or np.where(P < 0, 1, 0).any())):
        return None
    states = [P]
    current = P
    while np.where(current != 0, 0, 1).any():
        current = np.dot(P, current)
        if any(np.allclose(current, i) for i in states):
            return None
        states.append(current)
    while True:
        previous = current
        current = np.dot(P, current)
        if np.array_equal(current, previous):
            return current[0:1]
