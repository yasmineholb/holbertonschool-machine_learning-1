#!/usr/bin/env python3
"""Slice a numpy array along axes given a dictionary"""


def np_slice(matrix, axes={}):
    """Slice a numpy array along axes given a dictionary"""
    slices = [slice(None)] * (max(axes) + 1)
    for axis, slicey in axes.items():
        slices[axis] = slice(*slicey)
    return matrix[tuple(slices)]
