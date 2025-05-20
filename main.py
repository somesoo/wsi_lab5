from config import CONFIG
from function import generate_data
from neuron_network.mlp import MLP
from neuron_network.optimizer import SGD
from neuron_network.loss import mse
import numpy as np
import matplotlib.pyplot as plt
from regresja import run_regression_baseline
from adam import run_adam_optimization
from adam2 import Adam as AdamBackprop
import copy

def main():
    np.random.seed(CONFIG["seed"])
    X, y = generate_data(CONFIG["n_samples"], CONFIG["range"], CONFIG["seed"])
    X = X.reshape(1, -1)
    y = y.reshape(1, -1)
    layer_sizes = [1] + CONFIG["hidden_layers"] + [1]
    model_sgd = MLP(layer_sizes, activation=CONFIG["activation"])
    optimizer_SGD = SGD(learning_rate=CONFIG["learning_rate"])

    print("Start treningu...")
    model_sgd.train(X, y, epochs=CONFIG["epochs"], optimizer=optimizer_SGD)
    print("Trening zakończony.")

    y_pred = model_sgd.predict(X)
    final_loss = mse(y, y_pred)
    print(f"Błąd końcowy (MSE): {final_loss:.5f}")

    x_vals = X.flatten()
    true_vals = y.flatten()
    pred_vals = y_pred.flatten()

    sorted_idx = np.argsort(x_vals)
    x_vals = x_vals[sorted_idx]
    true_vals = true_vals[sorted_idx]
    pred_vals = pred_vals[sorted_idx]

    plt.figure(figsize=(10, 5))
    plt.plot(x_vals, true_vals, label="f(x) – funkcja rzeczywista", color="black")
    plt.plot(x_vals, pred_vals, label="MLP – aproksymator", linestyle="--")
    plt.title("Aproksymacja funkcji przez MLP")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # === MODELE DO PORÓWNANIA ===
    y_poly, loss_poly = run_regression_baseline(X, y)
#    model_adam = MLP(layer_sizes, activation=CONFIG["activation"])
    optimizer_adam = AdamBackprop()
    model_adam = MLP(layer_sizes, activation=CONFIG["activation"])
    model_adam.train(X, y, epochs=(CONFIG["epochs"] // 5 ), optimizer=optimizer_adam)
    y_pred_adam = model_adam.predict(X)
    loss_adam = mse(y, y_pred_adam)
    print(f"Błąd końcowy (MSE): {loss_adam:.5f}")
    print(f"Regresja (baseline) MSE: {loss_poly:.5f}")

    # === WYKRES PORÓWNAWCZY ===
    y_poly = y_poly.flatten()
    y_pred_adam = y_pred_adam.flatten()

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, true_vals, label="f(x) – funkcja rzeczywista", color="black")
    plt.plot(x_vals, pred_vals, label="MLP (SGD)", linestyle="--")
    plt.plot(x_vals, y_pred_adam[sorted_idx], label="MLP (ADAM)", linestyle="-.")
    plt.plot(x_vals, y_poly[sorted_idx], label="Regresja wielomianowa", linestyle=":")
    plt.title("Porównanie modeli aproksymujących")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()



# Najlepsza konfiguracja:
#  - Ilośc epok: 19500
#  - Learning rate: 0.7
#  - Liczba warstw: 5
#  - Neuronów w każdej warstwie: 100
#  - Końcowy błąd (MSE): 0.01255
#  - Czas treningu: 14.64 s
