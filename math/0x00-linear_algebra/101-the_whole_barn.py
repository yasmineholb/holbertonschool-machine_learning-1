#!/usr/bin/env python3
"""Elementwise add two n-dimensional matrices"""


def add_matrices(mat1, mat2):
    """Recursively construct a new sum of two matrices"""
    try:
        if (len(mat1) != len(mat2)):
            return None
        newmat = []
        for row1, row2 in zip(mat1, mat2):
            newrow = add_matrices(row1, row2)
            if newrow is None:
                return None
            newmat.append(newrow)
        return newmat
    except TypeError:
        """Should no longer be an iterable if we get here"""
        return mat1 + mat2
