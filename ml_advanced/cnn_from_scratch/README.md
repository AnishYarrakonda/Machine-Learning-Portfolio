# Convolutional Neural Network From Scratch

Implementation of a convolutional neural network built without high-level libraries, trained on FashionMNIST data.

## Components
- `cnn.py`: convolution helpers (`im2col`, `col2im`) plus the `Conv2D` layer.
- `pool.py`: pooling layers (max and average) with forward/backward implementations.
- `dense.py`, `activations.py`, `loss.py`, `model.py`, `optimizer.py`: shared dense/activation/loss/optimization building blocks reused from the fully connected project.
- `data/`: local copy of FashionMNIST (`raw/` contains `.idx` files) so that training can run offline.
- `test_mnist.py`: training script that constructs a CNN stack, runs epochs, and reports test accuracy.

## Running
1. Ensure the raw FashionMNIST files remain in `data/FashionMNIST/raw/`; they are not downloaded automatically.
2. Install dependencies if missing:
   ```bash
   pip install numpy matplotlib
   ```
3. Run the training script:
   ```bash
   cd machine_learning/ml_advanced/cnn_from_scratch
   python test_mnist.py
   ```

`test_mnist.py` logs loss/accuracy and saves interim plots to help visualize convergence.

## Notes
- The optimizer defaults to SGD with momentum but can be swapped to Adam/AdamW via keyword args inside `test_mnist.py`.
- Dropout layers, pooling, and dense layers interleave to mimic a textbook CNN architecture learned from first principles.
