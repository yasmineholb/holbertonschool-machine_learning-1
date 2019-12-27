#!/usr/bin/env python3
"""Calculate definiteness of a numpy matrix"""


import numpy as np


def definiteness(matrix):
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2:
        return None
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        return None
    if not np.allclose(matrix, matrix.T):
        return None
    # -1 means no definite entry yet
    definite = np.linalg.det(matrix)
    positive = -1
    negative = -1
    for i in range(matrix.shape[0]):
        determinant = np.linalg.det(matrix[:i + 1, :i + 1])
        if determinant != 0:
            if determinant > 0:
                if positive != 0:
                    positive = 1
                if negative != 0 and i % 2:
                    negative = 1
            else:
                positive = 0
                if negative != 0 and not i % 2:
                    negative = 1
                else:
                    negative = 0
    if definite:
        if positive:
            return "Positive definite"
        if negative:
            return "Negative definite"
    if positive:
        return "Positive semi-definite"
    if negative:
        return "Negative semi-definite"
    return "Indefinite"
