#!/usr/bin/env python3
"""Calculate derivative of list of polynomial coefficients"""


def poly_derivative(poly):
    """Calculate derivative of list of polynomial coefficients"""
    try:
        if len(poly) == 0:
            return None
    except TypeError:
        return None
    lastidx = 0
    newpoly = []
    for power, coef in enumerate(poly[1:]):
        if not (type(coef) is int or type(coef) is float):
            return None
        if coef != 0:
            lastidx = power
        newpoly.append(coef * (power + 1))
    if lastidx == 0:
        return [0]
    return newpoly[:lastidx + 1]
