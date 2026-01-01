"""Helper utilities for mean-field model configuration."""

import numpy as np


def gen_ext_arrays(original: np.ndarray | list,
                 n_populations: int,
                 n_clusters: int,):
    """Project external parameters into population-wise arrays.

    Args:
        original: Scalar, `(n_populations,)` vector, or `(2,)` array of
            excitatory/inhibitory values.
        n_populations: Total number of populations.
        n_clusters: Number of excitatory clusters (excluding background).

    Returns:
        An array of length `n_populations` with projected values.
    """
    original = np.asarray(original)
    if original.shape == (n_populations,):
        return original
    if original.ndim == 0:
        return np.repeat(original, n_populations)
    arr = np.zeros(n_populations)
    arr[:n_clusters + 1] = original[0]
    arr[n_clusters + 1:] = original[1]
    return arr
