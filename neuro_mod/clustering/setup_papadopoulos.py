from functools import lru_cache
from itertools import product, chain

import numpy as np


def setup_matrices(
        n_neurons: int,
        n_excitatory: int,
        n_excitatory_background: int,
        n_inhibitory_background: int,
        n_clusters: int,
        connectivity_fraction_probs: np.ndarray[float] | list[float],
        j_baseline: list[float] | np.ndarray[float],
        j_potentiated: list[float] | np.ndarray[float],
        potentiate: bool = True
):
    n_inhibitory = n_neurons - n_excitatory

    n_clustered_neurons = n_neurons - n_excitatory_background - n_inhibitory_background
    n_inhibitory_clustered = n_inhibitory - n_inhibitory_background
    n_excitatory_clustered = n_excitatory - n_excitatory_background

    n_excitatory_per_cluster = int(n_excitatory_clustered / n_clusters)
    n_inhibitory_per_cluster = int(n_inhibitory_clustered / n_clusters)

    fraction_e = n_excitatory_per_cluster / n_excitatory
    fraction_i = n_inhibitory_per_cluster / n_inhibitory

    boundaries = np.cumsum(
        [n_excitatory_per_cluster] * n_clusters +
        [n_excitatory_background] +
        [n_inhibitory_per_cluster] * n_clusters +
         [n_inhibitory_background]
    )
    boundaries = np.insert(boundaries, 0, 0).astype(np.uint16)

    j_mat = np.zeros((n_clusters * 2 + 2, n_clusters * 2 + 2))  # N E clusters + N I clusters +
    # + 2 background
    c_mat = np.zeros_like(j_mat)

    e_idx = slice(0, n_clusters)
    i_idx = slice(n_clusters + 1, n_clusters * 2 + 1)

    background_e_idx = n_clusters
    background_i_idx = 2 * n_clusters + 1

    if potentiate:

        depress_factors = [
            depress_formula(f_a, f_b, n_clusters, j_potentiated[i])
            for i, (f_a, f_b) in enumerate(product([fraction_e, fraction_i], repeat=2))
        ]
    else:
        depress_factors = [1.] * 4

    j_baseline = np.asarray(j_baseline)
    j_baseline = j_baseline / (n_neurons ** .5)

    block_shape = [n_clusters, n_clusters]
    iterator = product([e_idx, i_idx], repeat=2)
    f_iterator = product([fraction_e, fraction_i], repeat=2)
    # Main populations (no background)
    for i, (pair, f_pair) in enumerate(zip(iterator, f_iterator)):
        j_mat[pair] = _gen_block(block_shape, j_baseline[i], j_potentiated[i], depress_factors[i])
        c_mat[pair] = connectivity_fraction_probs[i]

    # Background populations
    # b <-> pop
    b2p = product([background_e_idx, background_i_idx], [e_idx, i_idx], repeat=1)
    p2b = product([e_idx, i_idx], [background_e_idx, background_i_idx], repeat=1)
    iterator = chain(b2p, p2b)
    for i, pair in enumerate(iterator):
        g = depress_factors[i % 4]
        j_mat[pair] = j_baseline[i % 4] * g
        c_mat[pair] = connectivity_fraction_probs[i % 4]

    # b to b
    iterator = product([background_e_idx, background_i_idx], repeat=2)
    for i, pair in enumerate(iterator):
        j_mat[pair] = j_baseline[i % 4]
        c_mat[pair] = connectivity_fraction_probs[i % 4]

    types = {}
    types['E'] = (0, n_excitatory - n_excitatory_background)
    types['Eb'] = (n_excitatory - n_excitatory_background, n_excitatory)
    types['I'] = (n_excitatory, n_neurons - n_excitatory_background)
    types['Ib'] = (n_neurons - n_excitatory_background, n_neurons)
    return c_mat, j_mat.T, boundaries, types


@lru_cache(4)
def depress_formula(f_a, f_b, p, jab):
    try:
        nominator = f_a + f_b - f_a * f_b * (p + jab)
        denominator = f_a + f_b - f_a * f_b * (p + 1.)
        return nominator / denominator
    except ZeroDivisionError:
        return 1.


def _gen_block(
        shape: np.ndarray | list[int] | tuple[int, int],
        baseline: float,
        potentiated: float,
        depress_factor: float,
):
    jmat = np.zeros(shape)
    jmat += depress_factor * baseline
    np.fill_diagonal(jmat, potentiated * baseline)
    return jmat

