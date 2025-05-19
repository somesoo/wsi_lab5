from config import CONFIG
from function import generate_data
from neuron_network.mlp import MLP
from neuron_network.optimizer import SGD
from neuron_network.loss import mse
import numpy as np
import matplotlib.pyplot as plt

def main():
    np.random.seed(CONFIG["seed"])
    X, y = generate_data(CONFIG["n_samples"], CONFIG["range"], CONFIG["seed"])
    X = X.reshape(1, -1) / 10
    y = y.reshape(1, -1)
    layer_sizes = [1] + CONFIG["hidden_layers"] + [1]
    model = MLP(layer_sizes, activation=CONFIG["activation"])
    optimizer = SGD(learning_rate=CONFIG["learning_rate"])

    print("Start treningu...")
    model.train(X, y, epochs=CONFIG["epochs"], optimizer=optimizer)
    print("Trening zakończony.")

    y_pred = model.predict(X)
    final_loss = mse(y, y_pred)
    print(f"Błąd końcowy (MSE): {final_loss:.5f}")

    x_vals = X.flatten()
    true_vals = y.flatten()
    pred_vals = y_pred.flatten()

    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, true_vals, label="f(x) – funkcja rzeczywista", color="black")
    plt.plot(x_vals, pred_vals, label="MLP – aproksymator", linestyle="--")
    plt.title("Aproksymacja funkcji przez MLP")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

    pass


if __name__ == "__main__":
    main()
