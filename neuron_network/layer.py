import numpy as np
from neuron_network.activations import sigmoid, sigmoid_derivative, tanh, tanh_derivative

class Layer:
    def __init__(self, input_size, output_size, activation="sigmoid"):
        self.weights = np.random.randn(output_size, input_size) * 0.1
        self.biases = np.zeros((output_size, 1))
        self.input = None
        self.z = None
        self.a = None

        if activation == "sigmoid":
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == "tanh":
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == "none":
            self.activation = lambda x: x
            self.activation_derivative = lambda x: 1
        else:
            raise ValueError("Unsupported activation function")


    def forward(self, input):
        self.input = input  
        self.z = np.dot(self.weights, input) + self.biases
        self.a = self.activation(self.z)
        return self.a


    def backward(self, output_gradient):
        activation_grad = self.activation_derivative(self.z)  # dσ/dz
        dz = output_gradient * activation_grad  # dL/dz

        dw = np.dot(dz, self.input.T) / self.input.shape[1] # dL/dW
        db = np.mean(dz, axis=1, keepdims=True)       # dL/db

        self.dweights = dw
        self.dbiases = db
        self.dw = self.dweights
        self.db = self.dbiases

#        print("dz mean:", np.mean(np.abs(dz)))
#        print("dw mean:", np.mean(np.abs(dw)))

        return np.dot(self.weights.T, dz)

