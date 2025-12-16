
import numpy as np

def classify_active_clusters(rates: np.ndarray, z_threshold: float = 2.):
    means = rates.mean(axis=1)
    stds = rates.std(axis=1)
    threshold = means + z_threshold * stds
    active_clusters = (rates.T > threshold).T
    return active_clusters
