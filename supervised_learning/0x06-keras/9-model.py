#!/usr/bin/env python3
"""Save and load a whole keras model"""


import tensorflow.keras as K


def save_model(network, filename):
    """Save a whole keras model"""
    network.save(filename)


def load_model(filename):
    """Load a whole keras model"""
    return K.models.load_model(filename)
