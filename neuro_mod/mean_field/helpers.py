
import numpy as np


def gen_ext_arrays(original: np.ndarray | list,
                 n_populations: int,
                 n_clusters: int,):
    original = np.asarray(original)
    if original.shape == (n_populations,):
        return original
    if original.ndim == 0:
        return np.repeat(original, n_populations)
    arr = np.zeros(n_populations)
    arr[:n_clusters + 1] = original[0]
    arr[n_clusters + 1:] = original[1]
    return arr
