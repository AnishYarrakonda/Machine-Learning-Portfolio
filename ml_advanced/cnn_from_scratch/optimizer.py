# imports
from abc import ABC, abstractmethod
import numpy as np


# optimizer interface
class Optimizer(ABC):

    @abstractmethod
    def update(self, layer):
        pass


# SGD with Momentum
class SGDMomentum(Optimizer):

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.state = {}

    def update(self, layer):
        key = id(layer)
        if key not in self.state:
            self.state[key] = {
                "v_weights": np.zeros_like(layer.weights),
                "v_biases": np.zeros_like(layer.biases),
            }

        v_weights = self.state[key]["v_weights"]
        v_biases = self.state[key]["v_biases"]

        v_weights = self.momentum * v_weights - self.lr * layer.d_weights
        v_biases = self.momentum * v_biases - self.lr * layer.d_biases

        layer.weights += v_weights
        layer.biases += v_biases

        self.state[key]["v_weights"] = v_weights
        self.state[key]["v_biases"] = v_biases


# Adam
class Adam(Optimizer):

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.state = {}

    def update(self, layer):
        key = id(layer)
        if key not in self.state:
            self.state[key] = {
                "m_weights": np.zeros_like(layer.weights),
                "m_biases": np.zeros_like(layer.biases),
                "v_weights": np.zeros_like(layer.weights),
                "v_biases": np.zeros_like(layer.biases),
                "t": 0,
            }

        s = self.state[key]
        s["t"] += 1

        s["m_weights"] = self.beta1 * s["m_weights"] + (1 - self.beta1) * layer.d_weights
        s["m_biases"] = self.beta1 * s["m_biases"] + (1 - self.beta1) * layer.d_biases
        s["v_weights"] = self.beta2 * s["v_weights"] + (1 - self.beta2) * (layer.d_weights ** 2)
        s["v_biases"] = self.beta2 * s["v_biases"] + (1 - self.beta2) * (layer.d_biases ** 2)

        m_weights_hat = s["m_weights"] / (1 - self.beta1 ** s["t"])
        m_biases_hat = s["m_biases"] / (1 - self.beta1 ** s["t"])
        v_weights_hat = s["v_weights"] / (1 - self.beta2 ** s["t"])
        v_biases_hat = s["v_biases"] / (1 - self.beta2 ** s["t"])

        layer.weights -= self.lr * m_weights_hat / (np.sqrt(v_weights_hat) + self.eps)
        layer.biases -= self.lr * m_biases_hat / (np.sqrt(v_biases_hat) + self.eps)


# AdamW (decoupled weight decay)
class AdamW(Optimizer):

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {}

    def update(self, layer):
        key = id(layer)
        if key not in self.state:
            self.state[key] = {
                "m_weights": np.zeros_like(layer.weights),
                "m_biases": np.zeros_like(layer.biases),
                "v_weights": np.zeros_like(layer.weights),
                "v_biases": np.zeros_like(layer.biases),
                "t": 0,
            }

        s = self.state[key]
        s["t"] += 1

        s["m_weights"] = self.beta1 * s["m_weights"] + (1 - self.beta1) * layer.d_weights
        s["m_biases"] = self.beta1 * s["m_biases"] + (1 - self.beta1) * layer.d_biases
        s["v_weights"] = self.beta2 * s["v_weights"] + (1 - self.beta2) * (layer.d_weights ** 2)
        s["v_biases"] = self.beta2 * s["v_biases"] + (1 - self.beta2) * (layer.d_biases ** 2)

        m_weights_hat = s["m_weights"] / (1 - self.beta1 ** s["t"])
        m_biases_hat = s["m_biases"] / (1 - self.beta1 ** s["t"])
        v_weights_hat = s["v_weights"] / (1 - self.beta2 ** s["t"])
        v_biases_hat = s["v_biases"] / (1 - self.beta2 ** s["t"])

        layer.weights *= (1 - self.lr * self.weight_decay)
        layer.weights -= self.lr * m_weights_hat / (np.sqrt(v_weights_hat) + self.eps)
        layer.biases -= self.lr * m_biases_hat / (np.sqrt(v_biases_hat) + self.eps)
