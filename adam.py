import numpy as np
from scipy.optimize import minimize
from neuron_network.loss import mse

def flatten_weights(layers):
    return np.concatenate([layer.weights.flatten() for layer in layers] +
                          [layer.biases.flatten() for layer in layers])

def unflatten_weights(layers, flat_params):
    offset = 0
    for layer in layers:
        w_shape = layer.weights.shape
        b_shape = layer.biases.shape
        w_size = np.prod(w_shape)
        b_size = np.prod(b_shape)
        layer.weights = flat_params[offset:offset + w_size].reshape(w_shape)
        offset += w_size
        layer.biases = flat_params[offset:offset + b_size].reshape(b_shape)
        offset += b_size

def run_adam_optimization(model, X, y, maxiter=1000):
    def loss_fn(flat_params):
        unflatten_weights(model.layers, flat_params)
        output = model.forward(X)
        return mse(y, output)

    init_params = flatten_weights(model.layers)
    result = minimize(loss_fn, init_params, method="L-BFGS-B", options={"maxiter": maxiter})
    unflatten_weights(model.layers, result.x)
    return model.predict(X), result.fun
