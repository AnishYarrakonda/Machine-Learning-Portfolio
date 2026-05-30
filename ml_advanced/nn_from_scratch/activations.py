# imports
from abc import ABC, abstractmethod
import numpy as np


# activation interface
class Activation(ABC):

    # must have a forward pass method
    @abstractmethod
    def forward(self, inputs):
        pass

    # must have a backpropagation method
    @abstractmethod
    def backward(self, error):
        pass


# Rectified Linear Unit Activation Function (ReLU)
class ReLU(Activation):

    # constructs a ReLU activation layer
    def __init__(self):
        self.inputs = None                      # cache inputs

    # forward pass (clips values at zero)
    def forward(self, inputs):
        self.inputs = inputs                    # cache inputs
        return np.maximum(0, inputs)            # return clipped values

    # backward pass
    def backward(self, error):
        d_inputs = error * (self.inputs > 0)    # apply piecewise derivative and multiply by error
        return d_inputs                         # pass error back

# Sigmoid Activation Function
class Sigmoid(Activation):

    # constructs a Sigmoid activation layer
    def __init__(self):
        self.inputs = None                      # cache inputs

    # forward pass (applies sigmoid function)
    def forward(self, inputs):
        self.inputs = inputs                    # cache inputs
        return 1 / (1 + np.exp(-inputs))        # apply the function and return outputs

    # backward pass
    def backward(self, error):
        sigmoid = 1 / (1 + np.exp(-self.inputs))        # calculate the sigmoid of the inputs
        d_inputs = error * sigmoid * (1 - sigmoid)      # take the derivative sig(1-sig) and multiply with error
        return d_inputs                                 # pass this back

# Hyperbolic Tangent Activation Function (tanh)
class Tanh(Activation):

    # constructs a tanh activation layer
    def __init__(self):
        self.inputs = None

    # forward pass (applies tanh function)
    def forward(self, inputs):
        self.inputs = inputs                    # cache inputs
        return np.tanh(inputs)                  # apply the function and return outputs

    # backward pass
    def backward(self, error):
        tanh = np.tanh(self.inputs)             # calculate the hyperbolic tangent
        return error * (1 - tanh ** 2)          # return error using derivative

# Softmax Activation Function (for logits in output layer only)
class Softmax(Activation):

    # constructs a Softmax layer
    def __init__(self):
        self.inputs = None                      # cache inputs

    # forward pass (normalizes logits into exponent-based scores)
    def forward(self, inputs):
        self.inputs = inputs                                            # cache raw input logits
        small_inputs = inputs - np.max(inputs, axis=1, keepdims=True)   # shift each row by its max value to prevent huge exponents
        exp_values = np.exp(small_inputs)                               # exponentiate shifted logits
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)   # apply our formula to return the probability distribution

    # backward pass *SOFTMAX is MOST IMPORTANT
    def backward(self, error):
        # create an empty array to store the gradients w.r.t inputs
        # shape matches the incoming gradient from the next layer
        d_inputs = np.empty_like(error)

        # iterate over each sample in the batch
        for i, (single_output, single_error) in enumerate(zip(self.forward(self.inputs), error)):
            # reshape the output for this sample into a column vector
            # required to compute the Jacobian matrix
            single_output = single_output.reshape(-1, 1)

            # compute the Jacobian matrix for the softmax function
            # J_ij = ∂(softmax_i)/∂(input_j)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # multiply the Jacobian by the incoming gradient (chain rule)
            d_inputs[i] = np.dot(jacobian, single_error)

        # return the gradient array for the entire batch
        return d_inputs

# Leaky ReLU (small negative values instead of 0)
class LeakyReLU(Activation):

    # constructs a Leaky ReLU layer
    def __init__(self, alpha=0.01):
        self.inputs = None                      # cache inputs
        self.alpha = alpha                      # slope

    # forward pass (decreases the magnitude of negative values)
    def forward(self, inputs):
        self.inputs = inputs                                        # cache inputs
        return np.where(inputs > 0, inputs, self.alpha * inputs)    # multiply negative numbers by alpha

    # backward pass
    def backward(self, error):
        d_inputs = np.where(self.inputs > 0, 1, self.alpha)         # apply piecewise derivative
        return error * d_inputs                                     # pass error back

# Dropout (randomly zeros neurons during training to prevent overfitting)
class Dropout(Activation):

    # constructs a Dropout layer
    def __init__(self, rate=0.3):
        self.rate = rate                        # fraction of neurons to drop (e.g. 0.3 = 30%)
        self.mask = None                        # cache the dropout mask for backprop

    # forward pass (drops random neurons and scales survivors to preserve expected value)
    def forward(self, inputs, training=True):
        if not training:                                                    # during inference, pass through unchanged
            return inputs
        self.mask = np.random.binomial(1, 1 - self.rate, inputs.shape)     # sample a binary mask (1 = keep, 0 = drop)
        self.mask = self.mask / (1 - self.rate)                             # inverted dropout: scale up survivors so expected value stays the same
        return inputs * self.mask                                           # zero out dropped neurons

    # backward pass
    def backward(self, error):
        return error * self.mask                # apply the same mask so dropped neurons get zero gradient

# Linear Activation Function (identity — used for regression output layers)
class Linear(Activation):

    # constructs a Linear activation layer
    def __init__(self):
        pass                                    # no state needed — output equals input

    # forward pass (passes inputs through unchanged)
    def forward(self, inputs):
        return inputs                           # identity function — no transformation

    # backward pass
    def backward(self, error):
        return error                            # derivative of identity is 1 — error passes straight through