import numpy as np

def load() -> (np.ndarray, np.ndarray):
    x = np.load('x.npy')
    y = np.load('y.npy')
    return x, y
