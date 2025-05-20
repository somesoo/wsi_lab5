import numpy as np
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layers):
        self.t += 1
        for i, layer in enumerate(layers):
            if i not in self.m:
                self.m[i] = {
                    "weights": np.zeros_like(layer.weights),
                    "biases": np.zeros_like(layer.biases)
                }
                self.v[i] = {
                    "weights": np.zeros_like(layer.weights),
                    "biases": np.zeros_like(layer.biases)
                }

            # Aktualizacja pierwszego momentu (średni gradient)
            self.m[i]["weights"] = self.beta1 * self.m[i]["weights"] + (1 - self.beta1) * layer.dw
            self.m[i]["biases"] = self.beta1 * self.m[i]["biases"] + (1 - self.beta1) * layer.db

            # Aktualizacja drugiego momentu (średni kwadrat gradientu)
            self.v[i]["weights"] = self.beta2 * self.v[i]["weights"] + (1 - self.beta2) * (layer.dw ** 2)
            self.v[i]["biases"] = self.beta2 * self.v[i]["biases"] + (1 - self.beta2) * (layer.db ** 2)

            # Korekcja biasu
            m_hat_w = self.m[i]["weights"] / (1 - self.beta1 ** self.t)
            m_hat_b = self.m[i]["biases"] / (1 - self.beta1 ** self.t)
            v_hat_w = self.v[i]["weights"] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v[i]["biases"] / (1 - self.beta2 ** self.t)

            # Aktualizacja wag
            layer.weights -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.eps)
            layer.biases -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)