# imports
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, make_friedman1, make_friedman2
from dense import Dense
from activations import ReLU, Tanh, Dropout, Linear
from loss import MSE, MAE
from model import NNModel
from helper import batch_generator


# ── Hyperparameters ────────────────────────────────────────────────────────────

# dataset — pick one:
#   'california'  → predict house prices from 8 geographic/demographic features
#                   why it's hard: highly nonlinear — log-income, location clusters, ocean proximity interactions
#                   linear regression R² ≈ 0.60, good MLP should reach R² ≈ 0.82+
#   'friedman1'   → y = 10sin(πx₁x₂) + 20(x₃−0.5)² + 10x₄ + 5x₅ + noise
#                   why it's hard: the true function is explicitly nonlinear and interaction-heavy
#                   linear regression R² ≈ 0.45, a good MLP should reach R² ≈ 0.90+
#   'friedman2'   → y = sqrt(x₁² + (x₂x₃ − 1/(x₂x₄))²) + noise
#                   why it's hard: involves division and square roots of interactions — the hardest of the three
#                   linear regression R² ≈ 0.10, a good MLP should reach R² ≈ 0.85+
DATASET       = 'friedman1'

# loss function — pick one:
#   MSE() → penalizes large errors heavily — good default for regression
#   MAE() → more robust to outliers — better for california housing which has price outliers
LOSS_FN       = MSE()

# architecture — add/remove entries to change depth, edit values to change width
HIDDEN_LAYERS = [256, 256, 128, 64]    # neurons in each hidden layer
DROPOUT_RATES = [0.3, 0.2]            # dropout rate after each hidden layer (len <= len(HIDDEN_LAYERS) - 1)

# training
EPOCHS        = 500                    # total number of training epochs
LR            = 0.005                  # lower lr than classification — regression loss surfaces are steeper
BATCH_SIZE    = 64                     # mini-batch size
MOMENTUM      = 0.9                    # momentum coefficient for weight updates
OPTIMIZER     = 'sgd_momentum'         # choose: 'sgd_momentum', 'adam', 'adamw'
VAL_SPLIT     = 0.8                    # fraction of data used for training
PRINT_EVERY   = 25                     # print stats every N epochs

if OPTIMIZER == 'sgd_momentum':
    OPTIMIZER_KWARGS = {'momentum': MOMENTUM}
elif OPTIMIZER == 'adamw':
    OPTIMIZER_KWARGS = {'weight_decay': 0.01}
else:
    OPTIMIZER_KWARGS = {}

# ──────────────────────────────────────────────────────────────────────────────


# helper function: computes R² (coefficient of determination) — 1.0 is perfect, 0.0 = no better than predicting the mean
def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)             # residual sum of squares
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)      # total sum of squares
    return 1 - ss_res / ss_tot                          # return R² score

# helper function: plots predicted vs actual values at the end of training
def plot_predictions(y_true, y_pred, title='Predicted vs Actual'):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')                        # scatter predicted vs actual
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect prediction')                 # diagonal = perfect line
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# helper function: plots training and validation loss curves for regression
def plot_loss_curve(train_losses, val_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses,   label='Val Loss',   color='orange')
    best_epoch = int(np.argmin(val_losses))
    best_loss  = val_losses[best_epoch]
    plt.axvline(x=best_epoch, color='green', linestyle='--',
                label=f'Best Val Loss Epoch {best_epoch + 1} ({best_loss:.4f})')
    plt.scatter(best_epoch, best_loss, color='green', zorder=5)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# load the selected dataset
if DATASET == 'california':
    data     = fetch_california_housing()
    X, y_raw = data.data.astype(float), data.target.astype(float)

elif DATASET == 'friedman1':
    # 10 input features, only 5 matter — tests whether the model can find the right interactions
    X, y_raw = make_friedman1(n_samples=5000, n_features=10, noise=1.0, random_state=42)
    X, y_raw = X.astype(float), y_raw.astype(float)

elif DATASET == 'friedman2':
    # 4 input features, all matter but in deeply nonlinear ways — hardest of the three
    X, y_raw = make_friedman2(n_samples=5000, noise=10.0, random_state=42)
    X, y_raw = X.astype(float), y_raw.astype(float)

else:
    raise ValueError(f"Unknown DATASET: '{DATASET}'. Choose from: 'california', 'friedman1', 'friedman2'")


# normalize features to zero mean and unit variance
scaler_X = StandardScaler()
X        = scaler_X.fit_transform(X)                        # normalize inputs

# normalize targets to zero mean and unit variance — critical for regression so MSE stays in a reasonable range
scaler_y = StandardScaler()
y        = scaler_y.fit_transform(y_raw.reshape(-1, 1))     # normalize targets
y        = y.ravel()                                        # flatten back to 1D

# shuffle data
indices  = np.arange(len(X))
np.random.shuffle(indices)
X, y     = X[indices], y[indices]

# split into train and validation sets
split         = int(VAL_SPLIT * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# reshape targets to (n, 1) — required for MSE/MAE which expect 2D arrays
y_train_2d = y_train.reshape(-1, 1)
y_val_2d   = y_val.reshape(-1, 1)

n_features = X.shape[1]                                    # input size comes from data

print(f'Dataset : {DATASET}')
print(f'Features: {n_features}  |  Samples: {len(X)}')
print(f'Train   : {len(X_train)}  |  Val: {len(X_val)}')
print('-' * 60)

# Build the model — Linear output activation so predictions are unbounded real values
model      = NNModel(
    lr=LR,
    optimizer=OPTIMIZER,
    optimizer_kwargs=OPTIMIZER_KWARGS
)
layer_sizes = [n_features] + HIDDEN_LAYERS                 # prepend actual feature count
for i, (n_in, n_out) in enumerate(zip(layer_sizes, layer_sizes[1:])):
    model.add(Dense(n_in, n_out))                          # add dense layer
    model.add(ReLU())                                      # ReLU works well for regression too
    start_idx = len(HIDDEN_LAYERS) - len(DROPOUT_RATES)    # find the starting layer to begin applying dropout
    if i >= start_idx:                                     # shift dropouts to later layers
        rate_idx = i - start_idx                           # find the corresponding dropout idx
        model.add(Dropout(DROPOUT_RATES[rate_idx]))        # apply dropout to prevent overfitting
model.add(Dense(HIDDEN_LAYERS[-1], 1))                     # output -> 1 (single predicted value)
model.add(Linear())                                        # linear output — no squashing, predictions are raw real numbers
model.set_loss(LOSS_FN)

# Training loop
train_loss_history, val_loss_history = [], []

for epoch in range(EPOCHS):
    # mini-batch training — iterate over shuffled batches each epoch
    model.training = True                                   # enable dropout during training
    for X_batch, y_batch in batch_generator(X_train, y_train_2d, BATCH_SIZE):
        pred = model.forward(X_batch)
        model.backward(pred, y_batch)

    # compute epoch-level metrics on full sets (no weight update)
    model.training = False                                  # disable dropout for evaluation
    y_pred     = model.forward(X_train)
    y_val_pred = model.forward(X_val)
    loss       = model.loss.forward(y_pred, y_train_2d)
    val_loss   = model.loss.forward(y_val_pred, y_val_2d)
    train_r2   = r_squared(y_train, y_pred.ravel())
    val_r2     = r_squared(y_val,   y_val_pred.ravel())
    train_loss_history.append(loss)
    val_loss_history.append(val_loss)

    # print stats every PRINT_EVERY epochs
    if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
        print(f'Epoch {epoch + 1:>4} / {EPOCHS} | '
              f'Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | '
              f'Train R²: {train_r2:.4f} | Val R²: {val_r2:.4f}')

# plot loss curves and predicted vs actual
plot_loss_curve(train_loss_history, val_loss_history)
plot_predictions(y_val, y_val_pred.ravel(),
                 title=f'{DATASET} — Val Predicted vs Actual (R²={val_r2:.3f})')
