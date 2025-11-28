
import numpy as np


def generate_clustered_weight_matrix(
        n_neurons: int,
        boundaries: list[float] | np.ndarray,
        synaptic_strengths: list[list[float]] | np.ndarray,
        connectivity: list[list[float]] | np.ndarray,
        random_generator: np.random.Generator = None,
        n_excitatory_background: int = 0,
        n_inhibitory_background: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a clustered m x m weight matrix for a spiking neuron model.

    Note: The standard convention W[i, j] is used, representing the connection
    strength from neuron j (pre-synaptic) to neuron i (post-synaptic).

    Args:
        n_neurons (int): Total number of neurons.
        boundaries (list[float]): the boundaries of each cluster.
        synaptic_strengths (list[list[float]]): A C x C matrix (list of lists)
            where extra_strengths[i][j] is the synaptic strength
            from cluster i (pre) to cluster j (post).
        connectivity (list[list[float]]): A C x C matrix (list of lists)
            the probability of a connection of a neuron from cluster i (pre) to cluster j (post) to form.
        random_generator (Optional[numpy.random.Generator], optional): Numpy random generator object.
         Defaults to None.
        n_excitatory_background (Optional[int], optional): Number of background excitatory
         neurons. Defaults to 0.
            number generator for reproducibility. Defaults to None.
        n_inhibitory_background (Optional[int], optional): Number of background inhibitory
         neurons. Defaults to 0.

    Returns:
        tuple[np.ndarray, np.ndarray]:
        - W (np.ndarray): The m x m weight matrix. W[i, j] is the
                          connection from j (pre) to i (post).
        - cluster_boundaries (np.ndarray): An array of shape (C, 2)
            where each row is the [start, end) index for a cluster.

    Raises:
        ValueError: If inputs are invalid (e.g., fractions don't sum to 1,
                    list lengths don't match C).


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

    p_matrix = np.zeros((n_neurons, n_neurons), dtype=np.bool)
    j_matrix = np.zeros((n_neurons, n_neurons))
    cluster_vec = np.empty(n_neurons, dtype=np.uint32)

    # --- 4. Create Full P and J matrices ---


    iterator = np.ndenumerate(connectivity)
    for (i, j), _ in iterator:
        b1 = slice(boundaries[i], boundaries[i + 1])
        b2 = slice(boundaries[j], boundaries[j + 1])
        size = b1.stop - b1.start, b2.stop - b2.start
        p_matrix[b1, b2] = (rng.random(size=size) <= connectivity[i, j]).astype(np.bool)
        j_matrix[b1, b2] = synaptic_strengths[i, j]
        if i <= j:
            cluster_vec[b2] = j + 1

    weights = p_matrix * j_matrix
    np.fill_diagonal(weights, 0)
    return weights, cluster_vec
