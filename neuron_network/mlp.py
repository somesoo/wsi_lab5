import numpy as np
from neuron_network.layer import Layer
from neuron_network.loss import mse, mse_derivative

class MLP:
    def __init__(self, layer_sizes, activation="sigmoid"):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1], activation))

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, loss_gradient, learning_rate):
        grad = loss_gradient
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward
            output = self.forward(X)

            # Loss
            loss = mse(y, output)

            # Backward
            grad = mse_derivative(y, output)
            self.backward(grad, learning_rate)

            # Można wypisać co jakiś czas:
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.5f}")

    def predict(self, X):
        return self.forward(X)
