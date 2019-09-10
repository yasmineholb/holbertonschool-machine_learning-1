#!/usr/bin/env python3
"""Transpose a list-matrix"""


def matrix_transpose(matrix):
    """Transpose a given list-matrix"""
    return [[row[idx] for row in matrix] for idx in range(len(matrix[0]))]
