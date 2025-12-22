
import numpy as np

def classify_active_clusters(rates: np.ndarray, threshold: float = .1):
    diffs = abs(np.diff(rates, axis=1)) / rates
    uniform = rates[diffs < threshold].mean()
    non_uniform = rates[diffs > threshold]
    if len(non_uniform) == 0:
        return uniform, uniform, uniform
    non_uniform_mean = np.mean(non_uniform)
    active_clustes_mean_rate = np.mean(non_uniform[non_uniform > non_uniform_mean])
    non_active_clustes_mean_rate = np.mean(non_uniform[non_uniform < non_uniform_mean])
    return active_clustes_mean_rate, non_active_clustes_mean_rate, uniform
