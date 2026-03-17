import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar

class WassExpSmooth:
    def __init__(self):
        self.theta = None

    def filter(self, theta, y):
        n, m = y.shape
        mu = np.empty((n, m))
        mu[0] = y[0]
        for i in range(1, n):
            mu[i] = theta * y[i - 1] + (1 - theta) * mu[i - 1]
        return mu

    def loss(self, theta, y):
        mu = self.filter(theta, y)
        return np.mean((y - mu) ** 2)

    def predict(self, y):
        if self.theta is None:
            raise Exception('Model not fitted!')
        return self.filter(self.theta, y)

    def fit(self, y):
        f = lambda theta: self.loss(theta, y)
        self.theta = minimize_scalar(
            f, bounds=(0, 1), method='bounded').x


class WassExpSmoothSampler:
    def __init__(self, theta, err_map, mu0=None):
        self.theta = theta
        self.err_map = err_map
        if mu0 is None:
            p = np.linspace(0.01, 0.99, 99)
            self.mu0 = norm.ppf(p)
        else:
            self.mu0=mu0

    def sample_err_map(self, n=50, a=-1.0, b=1.0, m=100):
        u = np.linspace(a, b, m)
        v = np.empty((n, m))
        for i in range(n):
            v[i] = self.err_map(u)
        return u, v

    def sample_path(self, n):
        m = len(self.mu0)
        mu = np.empty((n, m))
        y = np.empty((n, m))
        mu[0] = self.mu0
        y[0] = self.err_map(self.mu0)
        for i in range(1, n):
            mu[i] = self.theta * y[i - 1] + (1 - self.theta) * mu[i - 1]
            y[i] = self.err_map(mu[i])
        return (y, mu)

## EOF
