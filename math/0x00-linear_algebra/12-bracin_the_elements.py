#!/usr/bin/env python3
"""Do numpy matrix arithmetic"""


def np_elementwise(mat1, mat2):
    """Do numpy matrix arithmetic"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
