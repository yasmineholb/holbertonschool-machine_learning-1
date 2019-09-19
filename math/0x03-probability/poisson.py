#!/usr/bin/env python3
"""Poisson distribution calculations"""


class Poisson:
    """Poisson distribution stats class"""
    def __init__(self, data=None, lambtha=1.):
        """Initialize poisson distribution stats"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = lambtha
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """PMF at k number of events"""
        if k < 0:
            return 0
        k = int(k)
        return (pow(self.lambtha, k) *
                pow(2.7182818285, -1 * self.lambtha) /
                factorial(k))

    def cdf(self, k):
        """CDF at k number of events"""
        if k < 0:
            return 0
        k = int(k)
        return sum([self.pmf(n) for n in range(k + 1)])


def factorial(n):
    """Return factorial of n"""
    if n < 0:
        return None
    if n == 0:
        return 1
    if n < 2:
        return 1
    return n * factorial(n-1)
