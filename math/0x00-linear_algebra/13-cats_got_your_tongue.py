#!/usr/bin/env python3
"""Concat two numpy 2d matrices"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concat two numpy 2d matrices"""
    return np.append(mat1, mat2, axis=axis)
