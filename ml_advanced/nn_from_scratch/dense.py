# imports
import numpy as np

# Hidden Layer
class Dense:

    # constructs a hidden layer
    def __init__(self, num_inputs, num_outputs):
        self.inputs = None                                                                      # stores inputs for later
        self.weights = np.random.randn(num_inputs, num_outputs) * np.sqrt(1 / num_inputs)      # xavier initialization for tanh
        self.biases = np.zeros((1, num_outputs))                                                # biases are set to zero
        self.d_weights = None                                                                   # gradient of weights
        self.d_biases = None                                                                    # gradient of biases

    # forward pass
    def forward(self, inputs):
        self.inputs = inputs                                    # cache inputs
        outputs = inputs @ self.weights + self.biases           # calculate outputs using dot product and sum with biases
        return outputs                                          # return outputs for the next layer

    # backpropagation
    def backward(self, error):

        # gradient
        self.d_weights = self.inputs.T @ error                  # weights
        self.d_biases = np.sum(error, axis=0, keepdims=True)    # biases

        # create input for the previous layer
        d_inputs = error @ self.weights.T                       # backwards loss

        return d_inputs                                         # pass the loss back using chain rule
