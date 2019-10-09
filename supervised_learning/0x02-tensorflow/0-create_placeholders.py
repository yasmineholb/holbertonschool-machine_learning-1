#!/usr/bin/env python3
"""
Create Tensor placeholders for input data and one-hot labels
"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Create Tensor placeholders for input data and one-hot labels
    """
    return (tf.placeholder(float, shape=[None, nx], name='x'),
            tf.placeholder(float, shape=[None, classes], name='y'))
