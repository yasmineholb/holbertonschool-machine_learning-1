#!/usr/bin/env python3
"""Calculate a correlation matrix"""


import numpy as np


def correlation(C):
    """Calculate a correlation matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")
    corrmatrix = np.ndarray(C.shape)
    for row in range(C.shape[0]):
        for col in range(C.shape[1]):
            if row > col:
                corrmatrix[row][col] = corrmatrix[col][row]
                continue
            if col == row:
                corrmatrix[row][col] = 1
                continue
            corrmatrix[row][col] = C[row][col] / np.sqrt(C[row][row] *
                                                         C[col][col])
    return corrmatrix
