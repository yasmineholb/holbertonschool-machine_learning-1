#!/usr/bin/env python3
"""Determine probability of markov chain after some iterations"""


import numpy as np


def markov_chain(P, s, t=1):
    """
    Determine probability of markov chain after some iterations.
    P: transition matrix, (n, n) ndarray
    s: initial state, (1, n) ndarray
    t: number of iterations, int
    """
    if ((t < 1 or not isinstance(P, np.ndarray) or len(P.shape) != 2 or
         P.shape[0] < 1 or P.shape[0] != P.shape[1] or
         not isinstance(s, np.ndarray) or len(s.shape) != 2 or
         s.shape[1] != P.shape[0] or s.shape[0] != 1)):
        return None
    return np.linalg.multi_dot([s] + [P for i in range(t)])
