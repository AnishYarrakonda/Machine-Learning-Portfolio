# imports
from abc import ABC, abstractmethod
import numpy as np

# loss interface
class Loss(ABC):

    # must have a forward pass method
    @abstractmethod
    def forward(self, y_pred, y_true):
        pass

    # must have a backpropagation method
    @abstractmethod
    def backward(self, error, y_true):
        pass


# Mean Squared Error loss function
class MSE(Loss):

    # calculate loss
    def forward(self, y_pred, y_true):
        return np.mean((y_true - y_pred) ** 2)      # return squared error mean

    # backward pass
    def backward(self, y_pred, y_true):
        n = y_true.shape[0]                                 # number of samples in batch
        return 2 * (y_pred - y_true) / n                    # derivative of parabola is 2 * linear

# Mean Absolute Error loss function
class MAE(Loss):

    # calculate loss
    def forward(self, y_pred, y_true):
        return np.mean(np.abs(y_true - y_pred))             # return absolute error mean

    # backward pass
    def backward(self, y_pred, y_true):
        n = y_true.shape[0]                                 # number of samples in batch
        return np.where(y_pred - y_true > 0, 1, -1) / n     # apply piecewise derivative

# Binary Cross Entropy (expects sigmoid first, not raw logits)
class BCE(Loss):

    # calculate loss
    def forward(self, y_pred, y_true):
        eps = 1e-15                                                                     # small offset
        y_pred = np.clip(y_pred, eps, 1 - eps)                                          # clip to avoid log(0)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))    # return binary cross-entropy

    # backward pass
    def backward(self, y_pred, y_true):
        n = y_true.shape[0]                         # number of samples in batch
        return (y_pred - y_true) / n                # return derivative of BCE


# Categorical Cross Entropy (expects softmax first, not raw logits)
class CCE(Loss):

    # calculate loss
    def forward(self, y_pred, y_true):
        eps = 1e-15                                                 # small offset
        y_pred = np.clip(y_pred, eps, 1 - eps)                      # clip to avoid log(0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))    # return categorical cross-entropy

    # backward pass
    def backward(self, y_pred, y_true):
        eps = 1e-15                                 # small offset
        y_pred = np.clip(y_pred, eps, 1 - eps)      # clip to avoid log(0)
        n = y_true.shape[0]                         # number of samples in batch
        return -y_true / (y_pred * n)               # return true CCE gradient (let Softmax handle its own derivative)