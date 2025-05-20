import time
import numpy as np

from function import generate_data
from neuron_network.mlp import MLP
from neuron_network.optimizer import SGD
from neuron_network.loss import mse, mse_derivative

def optimize_architecture():
    learning_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    neuron_counts = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    layer_counts = [1, 2, 3, 4, 5]


    results = []

    # Dane wejściowe
    X, y = generate_data(n_samples=100, range=(-10, 10), seed=42)
    X = X.reshape(1, -1)
    y = y.reshape(1, -1)

    print("Start optymalizacji parametrów...\n")

    for lr in learning_rates:
        for num_layers in layer_counts:
            for neurons in neuron_counts:
                hidden_layers = [neurons] * num_layers

                model = MLP([1] + hidden_layers + [1], activation="tanh")
                optimizer = SGD(learning_rate=lr)
                epochs = 20000
                start = time.time()
                for epoch in range(epochs):  # ustalona liczba epok
                    y_pred = model.forward(X)
                    loss = mse(y, y_pred)
                    grad = mse_derivative(y, y_pred)
                    model.backward(grad)
                    optimizer.step(model.layers)
                    end = time.time()
                    if epoch % 500 == 0:
                        results.append({
                            "epoch": epoch,
                            "learning_rate": lr,
                            "layers": num_layers,
                            "neurons": neurons,
                            "loss": loss,
                            "time": end - start
                            })
                        print(f"{epoch},{num_layers},{neurons},{lr:.2f},{loss:.4f},{end-start:.2f}")

    # Sortowanie: najpierw jakość, potem czas
    best = sorted(results, key=lambda x: (x["loss"], x["time"]))[0]

    print("\nNajlepsza konfiguracja:")
    print(f" - Ilośc epok: {best['epoch']}")
    print(f" - Learning rate: {best['learning_rate']}")
    print(f" - Liczba warstw: {best['layers']}")
    print(f" - Neuronów w każdej warstwie: {best['neurons']}")
    print(f" - Końcowy błąd (MSE): {best['loss']:.5f}")
    print(f" - Czas treningu: {best['time']:.2f} s")

#    return best

optimize_architecture()
