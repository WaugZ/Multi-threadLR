import numpy as np


def sigmod(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, x, y, m):
    return (-y * np.log(sigmod(theta * x)) - (1 - y) * np.log(1 - sigmod(theta * x))) / m


def gradient(theta, x, y, m):
    dJ = x.T * (sigmod(theta * x) - y) / m
    return dJ

