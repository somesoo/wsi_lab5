import numpy as np


def f(x):
    return x**2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)


def generate_data(n_samples=1000, range=(-10, 10), seed=42):
    np.random.seed(seed)
    X = np.random.uniform(range[0], range[1], n_samples)
    y = f(X)
    return X, y