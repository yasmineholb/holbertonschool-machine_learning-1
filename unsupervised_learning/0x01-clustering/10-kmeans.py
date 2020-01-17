#!/usr/bin/env python3
"""Use sklearn to compute kmeans"""


import sklearn.cluster


def kmeans(X, k):
    """Use sklearn to compute kmeans"""
    means = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return means.cluster_centers_, means.labels_
