import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets

from activations import ReLU, Softmax
from cnn import Conv2D
from dense import Dense
from loss import CCE
from model import NNModel
from pool import MaxPool2D


# ── Hyperparameters ────────────────────────────────────────────────────────────

# data
DATA_DIR = "./data"
TRAIN_SAMPLES = 6000         # use a subset to keep pure-numpy CNN training practical
VAL_SAMPLES = 1000
TEST_SAMPLES = 2000
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5

# model architecture
CONV1_OUT_CHANNELS = 8
CONV2_OUT_CHANNELS = 16
KERNEL_SIZE = 3
PADDING = 1
POOL_SIZE = 2
POOL_STRIDE = 2
HIDDEN_UNITS = 256
N_CLASSES = 10

# training
EPOCHS = 10
LR = 0.001
BATCH_SIZE = 64
OPTIMIZER = "adam"
PRINT_EVERY = 1
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────────────────────

np.random.seed(RANDOM_SEED)


class Flatten:
    def __init__(self):
        self.input_shape = None

    def forward(self, inputs):
        self.input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, error):
        return error.reshape(self.input_shape)


def one_hot(y, num_classes):
    encoded = np.zeros((y.size, num_classes), dtype=np.float32)
    encoded[np.arange(y.size), y] = 1.0
    return encoded


def batch_generator(x, y, batch_size):
    indices = np.arange(x.shape[0])
    np.random.shuffle(indices)
    for start in range(0, x.shape[0], batch_size):
        batch_idx = indices[start:start + batch_size]
        yield x[batch_idx], y[batch_idx]


def evaluate(model, x, y_onehot, y_labels):
    model.training = False
    pred = model.forward(x)
    loss = model.loss.forward(pred, y_onehot)
    pred_classes = np.argmax(pred, axis=1)
    acc = np.mean(pred_classes == y_labels)
    return loss, acc


def plot_curves(train_loss, val_loss, train_acc, val_acc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(train_loss, label="Train Loss")
    axes[0].plot(val_loss, label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(train_acc, label="Train Acc")
    axes[1].plot(val_acc, label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.show()


def plot_random_predictions(model, x_test, y_test, class_names, n_samples=10):
    model.training = False
    idx = np.random.choice(x_test.shape[0], size=n_samples, replace=False)
    x_sample = x_test[idx]
    y_sample = y_test[idx]
    pred = model.forward(x_sample)
    pred_labels = np.argmax(pred, axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    axes = axes.flatten()
    for i in range(n_samples):
        is_correct = pred_labels[i] == y_sample[i]
        axes[i].imshow(x_sample[i, 0], cmap="gray")
        axes[i].set_title(
            f"P: {class_names[pred_labels[i]]}\nT: {class_names[y_sample[i]]}",
            color="green" if is_correct else "red",
            fontsize=9,
        )
        axes[i].axis("off")
    plt.suptitle("10 Random Fashion-MNIST Predictions (Green=Correct, Red=Incorrect)")
    plt.tight_layout()
    plt.show()


def main():
    train_ds = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True)
    test_ds = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True)
    class_names = train_ds.classes

    x_all = train_ds.data.numpy().astype(np.float32) / 255.0
    y_all = train_ds.targets.numpy().astype(np.int64)
    x_test_all = test_ds.data.numpy().astype(np.float32) / 255.0
    y_test_all = test_ds.targets.numpy().astype(np.int64)

    x_all = ((x_all - NORMALIZE_MEAN) / NORMALIZE_STD)[:, None, :, :]
    x_test_all = ((x_test_all - NORMALIZE_MEAN) / NORMALIZE_STD)[:, None, :, :]

    indices = np.random.permutation(x_all.shape[0])
    x_all, y_all = x_all[indices], y_all[indices]

    total_needed = TRAIN_SAMPLES + VAL_SAMPLES
    x_train = x_all[:TRAIN_SAMPLES]
    y_train = y_all[:TRAIN_SAMPLES]
    x_val = x_all[TRAIN_SAMPLES:total_needed]
    y_val = y_all[TRAIN_SAMPLES:total_needed]
    x_test = x_test_all[:TEST_SAMPLES]
    y_test = y_test_all[:TEST_SAMPLES]

    y_train_onehot = one_hot(y_train, N_CLASSES)
    y_val_onehot = one_hot(y_val, N_CLASSES)
    y_test_onehot = one_hot(y_test, N_CLASSES)

    model = NNModel(lr=LR, optimizer=OPTIMIZER)
    model.add(Conv2D(in_channels=1, out_channels=CONV1_OUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=POOL_SIZE, stride=POOL_STRIDE))
    model.add(Conv2D(in_channels=CONV1_OUT_CHANNELS, out_channels=CONV2_OUT_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING))
    model.add(ReLU())
    model.add(MaxPool2D(pool_size=POOL_SIZE, stride=POOL_STRIDE))
    model.add(Flatten())
    model.add(Dense(CONV2_OUT_CHANNELS * 7 * 7, HIDDEN_UNITS))
    model.add(ReLU())
    model.add(Dense(HIDDEN_UNITS, N_CLASSES))
    model.add(Softmax())
    model.set_loss(CCE())

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []

    for epoch in range(EPOCHS):
        model.training = True
        for xb, yb in batch_generator(x_train, y_train_onehot, BATCH_SIZE):
            y_pred = model.forward(xb)
            model.backward(y_pred, yb)

        train_loss, train_acc = evaluate(model, x_train, y_train_onehot, y_train)
        val_loss, val_acc = evaluate(model, x_val, y_val_onehot, y_val)
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)

        if epoch % PRINT_EVERY == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch + 1:>3}/{EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc * 100:.1f}% | Val Acc: {val_acc * 100:.1f}%"
            )

    test_loss, test_acc = evaluate(model, x_test, y_test_onehot, y_test)
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc * 100:.1f}%")

    plot_curves(train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist)
    plot_random_predictions(model, x_test, y_test, class_names, n_samples=10)


if __name__ == "__main__":
    main()
