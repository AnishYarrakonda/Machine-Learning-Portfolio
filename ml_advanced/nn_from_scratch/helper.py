#imports
import matplotlib.pyplot as plt
import numpy as np

# helper function: yields mini-batches
def batch_generator(X, y, batch_size):
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], y[batch_idx]


# helper function: plots the loss curve
def plot_accuracy_curve(train_acc, test_acc=None):
    """
    Plots training and optional testing accuracy curve,
    highlighting the best testing accuracy with a green vertical line.

    Parameters:
        train_acc: list or array of training accuracy values (0–1)
        test_acc: optional list or array of testing accuracy values (0–1)
    """
    train_acc = np.array(train_acc)
    plt.figure(figsize=(8, 5))

    # plot training accuracy
    plt.plot(train_acc, label='Train Accuracy', color='blue')

    if test_acc is not None:
        test_acc = np.array(test_acc)
        # plot testing accuracy
        plt.plot(test_acc, label='Test Accuracy', color='orange')

        # find best testing accuracy
        best_epoch = np.argmax(test_acc)
        best_acc = test_acc[best_epoch]

        # vertical line at best testing accuracy
        plt.axvline(x=best_epoch, color='green', linestyle='--',
                    label=f'Best Test Acc Epoch {best_epoch + 1} ({best_acc * 100:.2f}%)')
        plt.scatter(best_epoch, best_acc, color='green', zorder=5)

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


# helper function: plots decision boundary
def plot_decision_boundary(model, X_train, y_train, X_val=None, y_val=None, grid_step=0.005, ax=None, title=None, infer_batch_size=2048):
    """
    Plots the decision boundary of a trained model.

    Parameters:
        model: NNModel instance
        X_train, y_train: training data
        X_val, y_val: optional validation data
        grid_step: step size for meshgrid (smaller = sharper boundary but slower)
        ax: optional matplotlib Axes object
        title: optional plot title
        infer_batch_size: number of grid points to run through the model at once (reduces peak memory and keeps inference fast)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # create meshgrid
    x_min, x_max = X_train[:, 0].min() - 0.1, X_train[:, 0].max() + 0.1
    y_min, y_max = X_train[:, 1].min() - 0.1, X_train[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, grid_step),
                         np.arange(y_min, y_max, grid_step))

    # predict class for each point in meshgrid in batches to avoid slow single large forward pass
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = []
    for start in range(0, len(grid), infer_batch_size):
        batch  = grid[start:start + infer_batch_size]           # slice out the next chunk of grid points
        output = model.forward(batch)                           # run forward pass on this chunk only
        if output.shape[1] == 1:                                # binary output (BCE with single sigmoid neuron)
            preds.append((output.ravel() > 0.5).astype(int))   # threshold at 0.5 to get class 0 or 1
        else:                                                   # multi-class output (CCE / MSE / MAE with n_classes outputs)
            preds.append(np.argmax(output, axis=1))             # pick the highest scoring class
    Z = np.concatenate(preds).reshape(xx.shape)                 # stitch chunks back together and reshape to grid

    # plot contour
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

    # plot training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor='k', label='Train')

    # plot validation points if given
    if X_val is not None and y_val is not None:
        ax.scatter(X_val[:, 0], X_val[:, 1], c=y_val, edgecolor='k', marker='o', label='Val')

    # labels, title, legend
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    if title is not None:
        ax.set_title(title)
    ax.legend(loc='upper right')

    return ax