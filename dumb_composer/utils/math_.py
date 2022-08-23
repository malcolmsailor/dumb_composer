import numpy as np


def softmax(x):
    exp = np.exp(x)
    return exp / exp.sum()
