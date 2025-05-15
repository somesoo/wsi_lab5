import numpy as np
from scipy.special import expit  # dokładna implementacja sigmoida


# Sigmoid: 1 / (1 + exp(-x))
def sigmoid(x):
    return expit(x)


# Pochodna sigmoida: σ(x) * (1 - σ(x))
def sigmoid_derivative(x):
    s = expit(x)
    return s * (1 - s)


# Tanh: wbudowana w numpy
def tanh(x):
    return np.tanh(x)


# Pochodna tanh: 1 - tanh^2(x)
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2
