# imports
from make_data import generate_dataset
from dense import Dense
from activations import *
from loss import *
from model import NNModel
from helper import *


# ── Hyperparameters ────────────────────────────────────────────────────────────

# dataset
DATASET_TYPE = 'spiral'    # ['blobs', 'moons', 'circles', 'checkerboard', 'rings', 'spiral', 'wave', 'smiley']
POINTS       = 200         # points per class
CLASSES      = 5           # number of classes (for blobs, spiral, rings, etc.)
RADIUS       = 1.0         # max radius (for spiral/rings)
NOISE        = 0.1         # noise level for dataset
VAL_SPLIT    = 0.8         # fraction of data used for training (rest is validation)

# architecture — add/remove entries to change depth, edit values to change width
HIDDEN_LAYERS  = [256, 256, 128, 64]                        # neurons in each hidden layer
ACTIVATIONS    = [ReLU(), ReLU(), Tanh(), Tanh()]           # activation after each hidden layer (must match length of HIDDEN_LAYERS)
DROPOUT_RATES  = [0.3, 0.25, 0.2]                          # dropout rate after each hidden layer (len must be <= len(HIDDEN_LAYERS) - 1)

# loss and output — pick a compatible pair:
#   CCE  + Softmax  → multi-class classification  (n_classes outputs, one-hot labels)
#   BCE  + Sigmoid  → binary classification       (1 output, binary labels, CLASSES must be 2)
#   MSE  + Softmax  → multi-class regression loss (n_classes outputs, one-hot labels)
#   MAE  + Softmax  → multi-class regression loss (n_classes outputs, one-hot labels)
LOSS_FN = MSE()                 # loss function
OUTPUT_ACTIVATION = Softmax()   # output activation (applied after the final Dense layer)

# training
EPOCHS       = 1000                 # total number of training epochs
LR           = 0.02                 # base learning rate (momentum handles acceleration)
BATCH_SIZE   = 32                   # mini-batch size
MOMENTUM     = 0.9                  # momentum coefficient for weight updates
OPTIMIZER    = 'sgd_momentum'       # choose: 'sgd_momentum', 'adam', 'adamw'
PRINT_EVERY  = 50                   # print accuracy to console every N epochs

if OPTIMIZER == 'sgd_momentum':
    OPTIMIZER_KWARGS = {'momentum': MOMENTUM}
elif OPTIMIZER == 'adamw':
    OPTIMIZER_KWARGS = {'weight_decay': 0.01}
else:
    OPTIMIZER_KWARGS = {}

# ──────────────────────────────────────────────────────────────────────────────

# validate that ACTIVATIONS list matches HIDDEN_LAYERS
assert len(ACTIVATIONS) == len(HIDDEN_LAYERS), (
    f'ACTIVATIONS length ({len(ACTIVATIONS)}) must match HIDDEN_LAYERS length ({len(HIDDEN_LAYERS)})'
)

# Prepare data using the generate_dataset function
X, y = generate_dataset(
    dataset_type=DATASET_TYPE,
    n_samples=POINTS,
    n_classes=CLASSES,
    radius=RADIUS,
    noise=NOISE
)

# shuffle data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]

# split into train and validation sets
split = int(VAL_SPLIT * X.shape[0])
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# encode labels based on loss function
n_classes = int(y.max() + 1)                               # detect actual class count from data
if isinstance(LOSS_FN, BCE):
    # BCE expects raw binary labels — output is a single sigmoid neuron
    y_train_enc = y_train.reshape(-1, 1).astype(float)     # shape (n, 1)
    y_val_enc   = y_val.reshape(-1, 1).astype(float)       # shape (n, 1)
    n_outputs   = 1                                         # single output neuron for binary classification
else:
    # CCE / MSE / MAE all expect one-hot encoded labels
    y_train_enc = np.zeros((y_train.size, n_classes))
    y_train_enc[np.arange(y_train.size), y_train] = 1      # one-hot encode training labels
    y_val_enc   = np.zeros((y_val.size, n_classes))
    y_val_enc[np.arange(y_val.size), y_val]   = 1          # one-hot encode validation labels
    n_outputs   = n_classes                                 # one output neuron per class

# Build the model — loops through HIDDEN_LAYERS and ACTIVATIONS and DROPOUT_RATES automatically
model = NNModel(
    lr=LR,
    optimizer=OPTIMIZER,
    optimizer_kwargs=OPTIMIZER_KWARGS
)
layer_sizes = [2] + HIDDEN_LAYERS                           # insert input size at the front so we can zip adjacent pairs
for i, (n_in, n_out) in enumerate(zip(layer_sizes, layer_sizes[1:])):
    model.add(Dense(n_in, n_out))                           # add dense layer
    model.add(ACTIVATIONS[i])                               # add corresponding activation for this layer
    start_idx = len(HIDDEN_LAYERS) - len(DROPOUT_RATES)    # find the starting layer to begin applying dropout
    if i >= start_idx:                                      # shift dropouts to later layers
        rate_idx = i - start_idx                            # find the corresponding dropout idx
        model.add(Dropout(DROPOUT_RATES[rate_idx]))         # apply dropout to prevent overfitting
model.add(Dense(HIDDEN_LAYERS[-1], n_outputs))              # output -> n_outputs (no dropout before output)
model.add(OUTPUT_ACTIVATION)                                # apply output activation (e.g. Softmax or Sigmoid)
model.set_loss(LOSS_FN)

# Training loop
train_losses, val_losses = [], []
train_acc_history, val_acc_history = [], []

for epoch in range(EPOCHS):
    # mini-batch training — iterate over shuffled batches each epoch
    model.training = True                                   # enable dropout during training
    for X_batch, y_batch_enc in batch_generator(X_train, y_train_enc, BATCH_SIZE):
        y_batch_pred = model.forward(X_batch)
        model.backward(y_batch_pred, y_batch_enc)

    # compute epoch-level training loss on full train set (no weight update)
    model.training = False                                  # disable dropout for evaluation
    y_pred = model.forward(X_train)
    loss = model.loss.forward(y_pred, y_train_enc)
    train_losses.append(loss)

    # validation loss
    y_val_pred = model.forward(X_val)
    val_loss = model.loss.forward(y_val_pred, y_val_enc)
    val_losses.append(val_loss)

    # compute predicted classes — depends on output shape
    if isinstance(LOSS_FN, BCE):
        y_pred_class = (y_pred.ravel() > 0.5).astype(int)      # threshold sigmoid output at 0.5
        y_val_class  = (y_val_pred.ravel() > 0.5).astype(int)  # threshold sigmoid output at 0.5
    else:
        y_pred_class = np.argmax(y_pred, axis=1)                # pick highest scoring class
        y_val_class  = np.argmax(y_val_pred, axis=1)            # pick highest scoring class

    # compute accuracies
    train_acc = np.mean(y_pred_class == y_train)
    val_acc   = np.mean(y_val_class  == y_val)
    train_acc_history.append(train_acc)
    val_acc_history.append(val_acc)

    # print accuracy every PRINT_EVERY epochs
    if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
        print(f'Epoch {epoch + 1:>5} / {EPOCHS} | '
              f'Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'Train Acc: {train_acc * 100:.1f}% | Val Acc: {val_acc * 100:.1f}%')

# plot decision boundary once at the end
fig, ax = plt.subplots(figsize=(6, 6))
plot_decision_boundary(model, X_train, y_train, X_val, y_val, ax=ax, grid_step=0.005,
                       title=f'Final Decision Boundary | Train Acc: {train_acc_history[-1] * 100:.1f}% | Val Acc: {val_acc_history[-1] * 100:.1f}%')
plt.show()

# Plot training & validation accuracy
plot_accuracy_curve(train_acc_history, val_acc_history)
