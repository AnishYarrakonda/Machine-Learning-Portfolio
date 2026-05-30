# imports
from dense import Dense
from activations import *
from loss import *
from optimizer import Adam, AdamW, SGDMomentum, Optimizer

# base class for our neural network
class NNModel:

    # constructs a model (starts empty -> no layers)
    def __init__(self, lr=0.001, optimizer='sgd_momentum', optimizer_kwargs=None):
        self.layers = []                        # stores the layers
        self.loss = None                        # stores loss function
        self.lr = lr                            # stores learning rate
        self.optimizer = self._build_optimizer(optimizer, optimizer_kwargs or {})
        self.training = True                    # training mode flag (controls dropout behaviour)

    # builds optimizer from a name or optimizer instance
    def _build_optimizer(self, optimizer, optimizer_kwargs):
        if isinstance(optimizer, Optimizer):
            return optimizer

        key = optimizer.lower()
        if key in ('sgd', 'sgd_momentum', 'momentum'):
            return SGDMomentum(lr=self.lr, **optimizer_kwargs)
        if key == 'adam':
            return Adam(lr=self.lr, **optimizer_kwargs)
        if key == 'adamw':
            return AdamW(lr=self.lr, **optimizer_kwargs)
        raise ValueError(f"Unknown optimizer: '{optimizer}'. Choose from 'sgd_momentum', 'adam', 'adamw'.")

    # adds a layer
    def add(self, layer):
        self.layers.append(layer)               # add any layer type (dense, activation, etc.)

    # sets the loss function
    def set_loss(self, loss):
        self.loss = loss                        # use an appropriate loss function

    # forward pass
    def forward(self, X):
        for layer in self.layers:                       # loop through each layer
            if isinstance(layer, Dropout):              # dropout needs to know if we are training or not
                X = layer.forward(X, self.training)     # pass the training flag through
            else:                                       # otherwise
                X = layer.forward(X)                    # update the inputs through each layer's forward pass
        return X                                        # return the final outputs

    # backpropagation
    def backward(self, y_pred, y_true):
        error = self.loss.backward(y_pred, y_true)  # backwards (loss function)
        for layer in reversed(self.layers):         # go in reverse order of layers
            if isinstance(layer, Dense):            # if Dense layer
                error = layer.backward(error)       # compute gradients through Dense
                self.optimizer.update(layer)        # apply optimizer update
            else:                                   # otherwise
                error = layer.backward(error)       # just go backward
