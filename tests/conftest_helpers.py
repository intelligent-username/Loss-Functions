import numpy as np


def random_vectors(n_samples=5, n_features=4, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_samples, n_features))
