# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, make_hastie_10_2, make_classification
from sklearn.preprocessing import StandardScaler
from dense import Dense
from activations import ReLU, Tanh, Dropout, Softmax, Sigmoid
from loss import CCE, BCE
from model import NNModel
from helper import batch_generator, plot_accuracy_curve


# ── Dataset Selection ──────────────────────────────────────────────────────────
#
#   Three genuinely hard datasets — all require nonlinear boundaries that
#   logistic regression cannot learn:
#
#   'digits'      — 64 features (8x8 pixel handwritten digits), 10 classes
#                   logistic regression plateaus ~97%, NN can push past it
#
#   'hastie'      — 10 features, binary, explicitly designed to defeat linear
#                   models: y = sign(sum(x^2) - chi-squared median)
#
#   'complex'     — 25 features (15 informative, 5 redundant, 5 noise),
#                   5 classes, 4 clusters per class — highly nonlinear manifold
#
DATASET = 'digits'      # choose: 'digits', 'hastie', 'complex'

# ── Hyperparameters ────────────────────────────────────────────────────────────

# architecture — add/remove entries to change depth, edit values to change width
HIDDEN_LAYERS   = [256, 256, 128]       # neurons in each hidden layer
ACTIVATIONS     = [ReLU(), ReLU(), Tanh()]  # activation after each hidden layer (must match HIDDEN_LAYERS)
DROPOUT_RATES   = [0.3, 0.2]           # dropout after each hidden layer (len <= len(HIDDEN_LAYERS) - 1)

# training
EPOCHS          = 1000                  # total number of training epochs
LR              = 0.01                  # base learning rate
BATCH_SIZE      = 64                    # mini-batch size (larger for bigger datasets)
MOMENTUM        = 0.9                   # momentum coefficient
OPTIMIZER       = 'sgd_momentum'        # choose: 'sgd_momentum', 'adam', 'adamw'
VAL_SPLIT       = 0.8                   # fraction of data used for training
PRINT_EVERY     = 50                    # print stats every N epochs

if OPTIMIZER == 'sgd_momentum':
    OPTIMIZER_KWARGS = {'momentum': MOMENTUM}
elif OPTIMIZER == 'adamw':
    OPTIMIZER_KWARGS = {'weight_decay': 0.01}
else:
    OPTIMIZER_KWARGS = {}

# ──────────────────────────────────────────────────────────────────────────────


# load the selected dataset
if DATASET == 'digits':
    data = load_digits()
    X, y = data.data, data.target                   # 1797 samples, 64 features, 10 classes

elif DATASET == 'hastie':
    X, y = make_hastie_10_2(n_samples=12000, random_state=42)   # 12000 samples, 10 features, binary
    y = ((y + 1) / 2).astype(int)                               # remap labels from {-1, 1} to {0, 1}

elif DATASET == 'complex':
    X, y = make_classification(
        n_samples=3000,
        n_features=25,          # total features
        n_informative=15,       # features that actually carry signal
        n_redundant=5,          # linear combinations of informative features
        n_classes=5,            # 5-class problem
        n_clusters_per_class=4, # 4 gaussian clusters per class — very nonlinear
        random_state=42
    )

else:
    raise ValueError(f"Unknown DATASET: '{DATASET}'. Choose from: 'digits', 'hastie', 'complex'")

# normalize features to zero mean and unit variance — critical for deep networks on real data
scaler = StandardScaler()
X = scaler.fit_transform(X)                         # fit on all data before splitting

# shuffle data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

# split into train and validation sets
split = int(VAL_SPLIT * X.shape[0])
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# detect class count and encode labels
n_classes  = int(y.max() + 1)
n_features = X.shape[1]                             # input size comes from the data, not a hyperparameter

# one-hot encode for CCE
y_train_enc = np.zeros((y_train.size, n_classes))
y_train_enc[np.arange(y_train.size), y_train] = 1  # one-hot encode training labels
y_val_enc   = np.zeros((y_val.size, n_classes))
y_val_enc[np.arange(y_val.size), y_val]     = 1    # one-hot encode validation labels

print(f'\nDataset : {DATASET}')
print(f'Samples : {X.shape[0]}  (train={len(X_train)}, val={len(X_val)})')
print(f'Features: {n_features}')
print(f'Classes : {n_classes}\n')

# validate activation list length matches hidden layers
assert len(ACTIVATIONS) == len(HIDDEN_LAYERS), (
    f'ACTIVATIONS length ({len(ACTIVATIONS)}) must match HIDDEN_LAYERS length ({len(HIDDEN_LAYERS)})'
)

# Build the model — input size is driven by n_features, not hardcoded to 2
model = NNModel(
    lr=LR,
    optimizer=OPTIMIZER,
    optimizer_kwargs=OPTIMIZER_KWARGS
)
layer_sizes = [n_features] + HIDDEN_LAYERS              # prepend actual feature count
for i, (n_in, n_out) in enumerate(zip(layer_sizes, layer_sizes[1:])):
    model.add(Dense(n_in, n_out))                       # add dense layer
    model.add(ACTIVATIONS[i])                           # add corresponding activation
    start_idx = len(HIDDEN_LAYERS) - len(DROPOUT_RATES) # find the starting layer for dropout
    if i >= start_idx:                                  # shift dropouts to later layers
        rate_idx = i - start_idx                        # find the corresponding dropout index
        model.add(Dropout(DROPOUT_RATES[rate_idx]))     # apply dropout to prevent overfitting
model.add(Dense(HIDDEN_LAYERS[-1], n_classes))              # output -> n_classes
model.add(Softmax())
model.set_loss(CCE())

# Training loop
train_acc_history, val_acc_history = [], []

for epoch in range(EPOCHS):
    # mini-batch training — iterate over shuffled batches each epoch
    model.training = True                               # enable dropout during training
    for X_batch, y_batch_enc in batch_generator(X_train, y_train_enc, BATCH_SIZE):
        y_batch_pred = model.forward(X_batch)
        model.backward(y_batch_pred, y_batch_enc)

    # compute epoch-level metrics on full sets (no weight update)
    model.training = False                              # disable dropout for evaluation
    y_pred     = model.forward(X_train)
    y_val_pred = model.forward(X_val)
    loss       = model.loss.forward(y_pred, y_train_enc)
    val_loss   = model.loss.forward(y_val_pred, y_val_enc)

    # compute accuracies
    train_acc = np.mean(np.argmax(y_pred, axis=1) == y_train)
    val_acc   = np.mean(np.argmax(y_val_pred, axis=1) == y_val)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    # print every PRINT_EVERY epochs
    if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
        print(f'Epoch {epoch + 1:>5} / {EPOCHS} | '
              f'Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'Train Acc: {train_acc * 100:.1f}% | Val Acc: {val_acc * 100:.1f}%')

# plot accuracy curve — no decision boundary since we have more than 2 features
plot_accuracy_curve(train_acc_history, val_acc_history)
