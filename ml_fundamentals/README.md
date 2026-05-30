# Machine Learning Fundamentals

Notebook-based primer covering the core libraries, elementary regression models, and an applied PyTorch workflow.

## Directories
- `basic_ml_libraries/`: sequential notebooks for `numpy` (array ops, broadcasting, indexing), `pandas` (DataFrame manipulation), and plotting with Matplotlib/Seaborn (line/heatmap/dual-axis charts). Each subfolder also bundles sample CSV data used in that section.
- `basic_models/`: `linear_regression.ipynb` and `polynomial_regression.ipynb` show how to set up training loops, loss tracking, and plotting predictions versus ground truth.
- `pytorch/`: introduces tensors, data loading, model definition, training loops, and classification/computer vision examples. Includes `helper_functions.py`, reusable `models/`, prebuilt `data/`, and `practice.ipynb` for hands-on Q&A.

## Getting started
1. Create or activate a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn torch torchvision
   ```
3. Launch the notebook you want to explore, e.g.:
   ```bash
   cd machine_learning/ml_fundamentals
   jupyter-lab basic_ml_libraries/1\\ Numpy/00_numpy_fundamentals.ipynb
   ```

## Notes
- The PyTorch subfolder stores serialized helpers (`models/`) and a small `data/` directory so the notebooks can run offline.
- Use the regression notebooks to compare `sklearn` or custom gradient implementations before moving on to the from-scratch projects.
