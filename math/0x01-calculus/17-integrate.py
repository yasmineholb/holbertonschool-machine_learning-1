#!/usr/bin/env python3
"""Calculate integral of a polynomial from list of coefficients"""


def poly_integral(poly, C=0):
    """Calculate integral of a polynomial from list of coefficients"""
    try:
        if len(poly) < 1:
            return None
    except TypeError:
        return None
    if not (isinstance(C, int) or isinstance(C, float)):
        return None
    lastidx = 0
    for idx, coef in enumerate(poly):
        if not (type(coef) is int or type(coef) is float):
            return None
        if coef != 0:
            lastidx = idx + 1
    newlist = [C] + [int_if_whole(coef / (exp + 1))
                     for exp, coef in enumerate(poly)]
    return newlist[:lastidx + 1]


def int_if_whole(num):
    """Returns integer if number is whole, else returns number"""
    if num.is_integer():
        return int(num)
    else:
        return num
