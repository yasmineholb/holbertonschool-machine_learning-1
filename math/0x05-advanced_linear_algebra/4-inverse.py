#!/usr/bin/env python3
"""Find inverse matrix of a square list matrix"""


def inverse(matrix):
    """Find inverse matrix of a square list matrix"""
    origin_adjugate = adjugate(matrix)
    origin_deter = determinant(matrix)
    if origin_deter == 0:
        return None
    return sclr_mult_matrix(origin_adjugate, 1 / origin_deter)


def adjugate(matrix):
    """Find adjugate matrix of a square list matrix"""
    return matrix_transpose(cofactor(matrix))


def matrix_transpose(matrix):
    """Transpose a given list-matrix"""
    return [[row[idx] for row in matrix] for idx in range(len(matrix[0]))]


def cofactor(matrix):
    """Find cofactor matrix of a square list matrix"""
    minors = minor(matrix)
    return [[minors[row][col] * pow(-1, row) * pow(-1, col)
            for col in range(len(minors))] for row in range(len(minors))]


def minor(matrix):
    """Find minor matrix of a square list matrix"""
    if not isinstance(matrix, list) or len(matrix) < 1:
        raise TypeError("matrix must be a list of lists")
    for item in matrix:
        if not isinstance(item, list):
            raise TypeError("matrix must be a list of lists")
        if len(item) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    return [[determinant(slice_except(matrix, col, row))
             for col in range(len(matrix))]
            for row in range(len(matrix))]


def determinant(matrix):
    """
    Find determinant of a square list matrix.
    Check before recursion then return.
    """
    if not isinstance(matrix, list) or len(matrix) < 1:
        raise TypeError("matrix must be a list of lists")
    if ((len(matrix) == 1 and isinstance(matrix[0], list)
         and len(matrix[0]) == 0)):
        return 1
    for item in matrix:
        if not isinstance(item, list):
            raise TypeError("matrix must be a list of lists")
        if len(item) != len(matrix[0]) or len(item) != len(matrix):
            raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]
    return determinant_recur(matrix)


def determinant_recur(matrix):
    """
    Recursively find determinant of a matrix.
    """
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return sum([determinant_recur(slice_except(matrix, i, 0))
                * pow(-1, i) * matrix[0][i]
                for i in range(len(matrix))])


def slice_except(matrix, x, y):
    """Return a slice of a list of lists except a col and row"""
    return [[col for excol, col in enumerate(row) if excol != x]
            for exrow, row in enumerate(matrix) if exrow != y]


def sclr_mult_matrix(matrix, scalar):
    """Multiply matrix by a scalar"""
    return [[col * scalar for col in row] for row in matrix]
