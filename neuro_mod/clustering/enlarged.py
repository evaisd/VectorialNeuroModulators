"""Clustered connectivity matrix construction utilities."""

import numpy as np


def generate_clustered_weight_matrix(
        n_neurons: int,
        boundaries: list[float] | np.ndarray,
        synaptic_strengths: list[list[float]] | np.ndarray,
        connectivity: list[list[float]] | np.ndarray,
        random_generator: np.random.Generator = None,
        n_excitatory_background: int = 0,
        n_inhibitory_background: int = 0,
        types: dict[int, tuple[int, int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a clustered weight matrix for spiking neuron models.

    Note:
        The convention `W[i, j]` is used to represent the connection strength
        from neuron `j` (pre-synaptic) to neuron `i` (post-synaptic).

    Args:
        n_neurons: Total number of neurons.
        boundaries: Cluster boundary indices, length `C + 1`.
        synaptic_strengths: `(C, C)` matrix of synaptic strengths where
            `synaptic_strengths[i, j]` is from cluster `i` (pre) to `j` (post).
        connectivity: `(C, C)` matrix of connection probabilities.
        random_generator: Optional NumPy random generator.
        n_excitatory_background: Number of background excitatory neurons.
        n_inhibitory_background: Number of background inhibitory neurons.
        types: Optional cluster type boundaries (unused, for compatibility).

    Returns:
        A tuple `(weights, cluster_vec)` where:
        - `weights` is the `(n_neurons, n_neurons)` weight matrix.
        - `cluster_vec` assigns each neuron to its cluster id.

    Raises:
        ValueError: If `connectivity` or `synaptic_strengths` do not match
            the expected `(n_clusters, n_clusters)` shape.
    """

    # --- 0. Set random seed for reproducibility ---

    rng = np.random.default_rng(256) if random_generator is None else random_generator

    # --- 1. Move to NumPy ---

    synaptic_strengths = np.asarray(synaptic_strengths)
    connectivity = np.asarray(connectivity)

    # --- 2. Validate Inputs ---
    n_clusters = len(boundaries) - bool(n_excitatory_background) - bool(n_inhibitory_background) + 1

    desired_shape = (n_clusters, n_clusters)

    if not all(item.shape == desired_shape for item in [
        synaptic_strengths,
        connectivity
    ]):
        raise ValueError("The connectivity and strengths must be n_clusters x n_clusters "
                         "matrices (or list of lists)")

    # --- 3. Initiate Full Matrices and Assign Clusters ---

    p_matrix = np.zeros((n_neurons, n_neurons), dtype=bool)
    j_matrix = np.zeros((n_neurons, n_neurons), dtype=np.float64)
    cluster_vec = np.empty(n_neurons, dtype=np.uint8)

    # --- 4. Create Full P and J matrices ---


    iterator = np.ndenumerate(connectivity)
    for (i, j), _ in iterator:
        b1 = slice(boundaries[i], boundaries[i + 1])
        b2 = slice(boundaries[j], boundaries[j + 1])
        size = b1.stop - b1.start, b2.stop - b2.start
        p_matrix[b1, b2] = _generate_equirowsum_matrix(*size, connectivity[i, j], rng).astype(bool)
        # p_matrix[b1, b2] = (rng.random(size=size) <= connectivity[i, j]).astype(np.bool)
        j_matrix[b1, b2] = synaptic_strengths[i, j]
        if i <= j:
            cluster_vec[b2] = j + 1

    weights = p_matrix * j_matrix
    np.fill_diagonal(weights, 0)
    return weights, cluster_vec


def _generate_equirowsum_matrix(n, m, p, rng: np.random.Generator):
    k = int(p * m)
    base = np.zeros(m, dtype=np.uint8)
    base[:k] = 1
    mat = np.tile(base, (n, 1))
    keys = rng.random([n, m])
    mat = mat[np.arange(n)[:, None], np.argsort(keys, axis=1)]
    return mat
