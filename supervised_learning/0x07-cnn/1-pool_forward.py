#!/usr/bin/env python3
"""Convlutional forward propagation, manually"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """Convlutional forward propagation, manually"""
    bottom = A_prev.shape[1] - kernel_shape[0] + 1
    right = A_prev.shape[2] - kernel_shape[1] + 1
    output = np.ndarray((A_prev.shape[0], int((bottom - 1) / stride[0] + 1),
                         int((right - 1) / stride[1] + 1), A_prev.shape[3]))
    y_in = 0
    y_out = 0
    while y_in < bottom:
        x_in = 0
        x_out = 0
        while x_in < right:
            slice = A_prev[:, y_in:y_in + kernel_shape[0],
                           x_in:x_in + kernel_shape[1], :]
            if mode == 'max':
                output[:, y_out, x_out] = np.amax(slice, axis=(1, 2))
            else:
                output[:, y_out, x_out] = np.mean(slice, axis=(1, 2))
            x_in += stride[1]
            x_out += 1
        y_in += stride[0]
        y_out += 1
    return output
