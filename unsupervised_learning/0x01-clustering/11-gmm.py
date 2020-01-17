#!/usr/bin/env python3
"""Use sklearn to compute gaussian mixture model"""


import sklearn.mixture


def gmm(X, k):
    """Use sklearn to compute gaussian mixture model"""
    mixture = sklearn.mixture.GaussianMixture(k).fit(X)
    return (mixture.weights_, mixture.means_, mixture.covariances_,
            mixture.predict(X), mixture.bic(X))
