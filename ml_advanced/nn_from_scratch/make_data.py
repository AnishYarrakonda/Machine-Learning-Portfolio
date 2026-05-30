import numpy as np
from sklearn.datasets import make_blobs, make_moons, make_circles


def generate_dataset(dataset_type: str, n_samples=300, noise=0.05, n_classes=2, radius=1.0, grid_size=2):
    """
    Generates various 2D classification datasets for neural network experiments.

    Parameters:
        dataset_type: str, one of ['blobs', 'moons', 'circles', 'checkerboard', 'rings', 'spiral', 'wave', 'smiley']
        n_samples: number of points PER CLASS (total points = n_samples * n_classes)
        noise: float, noise level
        n_classes: int, number of classes (for blobs, rings, spiral)
        radius: float, maximum radius (for spiral/rings)
        grid_size: int, grid size for checkerboard

    Returns:
        X: np.ndarray of shape (n_samples * n_classes, 2), normalized to [-1, 1]
        y: np.ndarray of shape (n_samples * n_classes,)
    """
    total = n_samples * n_classes                   # total points across all classes

    if dataset_type == 'blobs':
        X, y = make_blobs(n_samples=total, centers=n_classes, cluster_std=noise * 2, random_state=42)

    elif dataset_type == 'moons':
        X, y = make_moons(n_samples=total, noise=noise, random_state=42)

    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=total, noise=noise, factor=0.5)

    elif dataset_type == 'checkerboard':
        X = np.random.rand(total, 2) * grid_size
        y = ((X[:, 0] // 1 + X[:, 1] // 1) % 2).astype(int)

    elif dataset_type == 'rings' or dataset_type == 'concentric_rings':
        # Generate multiple concentric rings
        X_list = []
        y_list = []
        for i in range(n_classes):
            r = i + np.random.rand(n_samples) * noise + 0.5    # radius for this ring
            theta = np.random.rand(n_samples) * 2 * np.pi
            x = r * np.cos(theta)
            y_ = r * np.sin(theta)
            X_list.append(np.c_[x, y_])
            y_list.append(np.full(n_samples, i))
        X = np.vstack(X_list)
        y = np.hstack(y_list)

    elif dataset_type == 'spiral':
        X = np.zeros((total, 2))
        y = np.zeros(total, dtype='uint8')
        for class_number in range(n_classes):
            ix = range(n_samples * class_number, n_samples * (class_number + 1))
            r = np.linspace(0.0, radius, n_samples)
            t = np.linspace(class_number * 4, (class_number + 1) * 4, n_samples) + np.random.randn(n_samples) * noise
            X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
            y[ix] = class_number

    elif dataset_type == 'wave':
        # two classes separated by a sine wave boundary — hard for linear models, easy to visualise
        X = np.random.rand(total, 2)                            # generate in [0, 1] first, normalize later
        boundary = np.sin(X[:, 0] * 2 * np.pi) * 0.2          # sine wave boundary in unit space
        y = (X[:, 1] > boundary + 0.5 + np.random.randn(total) * noise).astype(int)

    elif dataset_type == 'smiley':
        # a two-class face: class 0 = eyes + mouth arc, class 1 = face outline circle
        n_each = n_samples // 2                                 # split n_samples evenly across face features

        # face outline (class 1) — n_samples points total on the outer ring
        theta_face = np.random.rand(n_samples) * 2 * np.pi
        r_face = 1.0 + np.random.randn(n_samples) * noise
        face = np.c_[r_face * np.cos(theta_face), r_face * np.sin(theta_face)]

        # left eye (class 0)
        theta_eye = np.random.rand(n_each) * 2 * np.pi
        r_eye = 0.15 + np.random.randn(n_each) * noise
        left_eye = np.c_[r_eye * np.cos(theta_eye) - 0.35, r_eye * np.sin(theta_eye) + 0.35]

        # right eye (class 0)
        right_eye = np.c_[r_eye * np.cos(theta_eye) + 0.35, r_eye * np.sin(theta_eye) + 0.35]

        # mouth arc (class 0) — bottom semicircle, remaining points
        n_mouth = n_samples - 2 * n_each
        theta_mouth = np.linspace(np.pi + 0.3, 2 * np.pi - 0.3, n_mouth)
        r_mouth = 0.5 + np.random.randn(n_mouth) * noise
        mouth = np.c_[r_mouth * np.cos(theta_mouth), r_mouth * np.sin(theta_mouth) + 0.1]

        X = np.vstack([face, left_eye, right_eye, mouth])
        y = np.hstack([np.ones(n_samples), np.zeros(n_each), np.zeros(n_each), np.zeros(n_mouth)]).astype(int)

    else:
        raise ValueError(f"Unknown dataset_type: '{dataset_type}'. "
                         f"Choose from: 'blobs', 'moons', 'circles', 'checkerboard', "
                         f"'rings', 'spiral', 'wave', 'smiley'")

    # normalize both axes to [-1, 1] so all datasets have the same bounding box
    X = X - X.min(axis=0)                          # shift so min is 0
    X = X / X.max(axis=0)                          # scale so max is 1
    X = X * 2 - 1                                  # remap to [-1, 1]

    return X, y