#!/usr/bin/env python3
"""Implement a modified LeNet-5 architecture using Tensorflow"""


import tensorflow as tf


def lenet5(X, Y):
    """Implement a modified LeNet-5 architecture using Tensorflow"""
    initializer = (tf.contrib.layers.
                   variance_scaling_initializer())
    out = tf.layers.Conv2D(6, 5, padding='same',
                           activation='relu',
                           kernel_initializer=initializer)(X)
    out = tf.layers.MaxPooling2D(2, 2)(out)
    out = tf.layers.Conv2D(16, 5, padding='valid', activation='relu',
                           kernel_initializer=initializer)(out)
    out = tf.layers.MaxPooling2D(2, 2)(out)
    out = tf.layers.Flatten()(out)
    out = tf.layers.Dense(120, activation='relu',
                          kernel_initializer=initializer)(out)
    out = tf.layers.Dense(84, activation='relu',
                          kernel_initializer=initializer)(out)
    out = tf.layers.Dense(10, activation='softmax',
                          kernel_initializer=initializer)(out)
    loss = tf.losses.softmax_cross_entropy(Y, out)
    train = tf.train.AdamOptimizer().minimize(loss)
    max_pred = tf.argmax(out, 1)
    equal = tf.equal(tf.argmax(Y, 1), max_pred)
    accuracy = tf.reduce_mean(tf.cast(equal, tf.float32))
    return out, train, loss, accuracy
