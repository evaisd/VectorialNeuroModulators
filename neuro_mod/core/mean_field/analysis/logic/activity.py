"""Activity classification utilities for mean-field rates."""

import numpy as np


def classify_active_clusters(rates: np.ndarray, threshold: float = .1):
    """Classify active and inactive cluster rates based on variability.

    Args:
        rates: Array of firing rates with populations along axis 1.
        threshold: Relative difference threshold for non-uniform activity.

    Returns:
        Tuple `(active_mean, inactive_mean, uniform_mean)` for the detected
        active clusters, inactive clusters, and uniform activity baseline.
    """
    diffs = abs(np.diff(rates, axis=1)) / rates
    uniform = rates[diffs < threshold].mean()
    non_uniform = rates[diffs > threshold]
    if len(non_uniform) == 0:
        return uniform, uniform, uniform
    non_uniform_mean = np.mean(non_uniform)
    active_clustes_mean_rate = np.mean(non_uniform[non_uniform > non_uniform_mean])
    non_active_clustes_mean_rate = np.mean(non_uniform[non_uniform < non_uniform_mean])
    return active_clustes_mean_rate, non_active_clustes_mean_rate, uniform
