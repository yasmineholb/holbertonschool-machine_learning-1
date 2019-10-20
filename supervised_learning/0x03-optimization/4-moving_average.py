#!/usr/bin/env python3
"""Calculate weighted moving average of data set"""


def moving_average(data, beta):
    """Calculate weighted moving average of data set"""
    ema = [0]
    unbias = []
    for idx, dat in enumerate(data):
        ema.append(beta * ema[idx] + (1 - beta) * dat)
        unbias.append(ema[idx + 1] / (1 - beta ** (idx + 1)))
    return unbias
