import numpy as np

def err_map_shift(x, s=1):
    b = np.random.normal(0, s)
    return x + b

def err_map_sine(x, s=2, a=0.3, k=3):
    c = np.random.normal(0, s, size=(k, 1))
    c = np.tile(c, (1, len(x)))
    x = np.tile(x, (k, 1))
    y = x - a * np.sin(np.pi * (x - c)) / (np.pi)
    w = np.random.uniform(size=(1, k))
    w = w / np.sum(w)
    return np.dot(w, y)

## EOF
