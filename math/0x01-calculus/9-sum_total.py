#!/usr/bin/env python3
"""Calculate summation of i^2 from 1 to n"""


def summation_i_squared(n):
    """Calculate summation of i^2 from 1 to n"""
    if type(n) is not int or n < 1:
        return None
    return int(n / 6 + n * n / 2 + pow(n, 3) / 3)
