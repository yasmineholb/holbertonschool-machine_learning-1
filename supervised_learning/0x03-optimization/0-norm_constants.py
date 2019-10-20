#!/usr/bin/env python3
"""Calculate normalization constants of a matrix"""


import numpy as np


def normalization_constants(X):
    """Calculate normalization constants of a matrix"""
    return X.mean(axis=0), X.std(axis=0)
