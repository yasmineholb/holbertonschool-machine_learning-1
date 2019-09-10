#!/usr/bin/env python3
"""Manually concat two ndimensional list-matrices"""


def cat_matrices(mat1, mat2, axis=0):
    """Manually concat two ndimensional list-matrices"""
    ptr1 = mat1
    ptr2 = mat2
    check_axis = 0
    try:
        while(True):
            if type(ptr1) != type(ptr2):
                return None
            if check_axis != axis and len(ptr1) != len(ptr2):
                return None
            check_axis = check_axis + 1
            ptr1 = ptr1[0]
            ptr2 = ptr2[0]
    except TypeError:
        pass
    return cat_matrices_recur(mat1, mat2, axis)


def cat_matrices_recur(mat1, mat2, axis):
    """Recursive concatenation of two matrices verified to be concattable"""
    if axis > 0:
        return [cat_matrices(row1, row2, axis - 1)
                for row1, row2 in zip(mat1, mat2)]
    return deepcopy(mat1) + deepcopy(mat2)


def deepcopy(listy):
    "Deep copy two list-like objects"
    try:
        return [deepcopy(item) for item in listy]
    except (TypeError, AttributeError):
        return listy
