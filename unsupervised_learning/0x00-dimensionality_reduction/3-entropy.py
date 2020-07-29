#!/usr/bin/env python3
"""Calculate Shannon entropies and P affinities relative to a data point."""


import numpy as np


def HP(Di, beta):
    """Calculate Shannon entropies and P affinities relative to a point."""
    exponent = np.exp(-Di/beta)
    Pi = exponent / exponent.sum()
    Hi = -(Pi*np.log2(Pi)).sum()
    return Hi, Pi
