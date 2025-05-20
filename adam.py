
import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam
from neuron_network.loss import mse
from neuron_network.mlp import MLP

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

def run_adam_optimization(model, X, y, maxiter=20000, step_size=0.01):
    def loss_fn(flat_params, iter):
        unflatten_weights(model.layers, flat_params)
        output = model.forward(X)
        return mse(y, output)

    init_params = flatten_weights(model.layers)
    loss_grad = grad(loss_fn)

    optimized_params = adam(loss_grad, init_params, step_size=step_size, num_iters=maxiter)
    unflatten_weights(model.layers, optimized_params)
    y_pred = model.predict(X)
    loss_value = mse(y, y_pred)
    return y_pred, loss_value