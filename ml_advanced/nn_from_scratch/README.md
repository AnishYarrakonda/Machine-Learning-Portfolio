# Neural Network From Scratch

Minimal neural-network framework built from first principles, covering dense layers, activations, loss functions, and optimizers.

## Highlights
- `dense.py`: fully connected layers with forward/backward passes and gradient storage.
- `activations.py`: ReLU, sigmoid, softmax, dropout helpers that respect training mode.
- `loss.py`: implementations of mean squared error and categorical cross entropy with gradient callbacks.
- `optimizer.py`: SGD with momentum plus Adam/AdamW, which the model wires up based on the selected `optimizer` argument.
- `model.py`: composable `NNModel` that accepts any sequence of layers, handles training mode, and delegates optimization to the configured optimizer.
- `helper.py` + `make_data.py`: utility scripts for generating toy datasets (patterns, blobs) used across the tests.

## Tests
- `test_sk_regression.py`: ensures the custom network matches `scikit-learn` on regression tasks.
- `test_sk_classification.py`: compares classification outputs against `sklearn` benchmarks.
- `test_patterns.py`: visual sanity checks on synthetic puzzles.

Run them with `pytest`:

```bash
cd machine_learning/ml_advanced/nn_from_scratch
python -m pytest test_sk_classification.py test_sk_regression.py test_patterns.py
```

## Notes
- Defaults use learning rate `0.001` with `SGDMomentum`.
- Extend the library by adding new layers (e.g., convolutional) or swapping the optimizer object when calling `NNModel(lr=..., optimizer=...)`.
