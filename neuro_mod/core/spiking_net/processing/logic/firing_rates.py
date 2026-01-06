"""Firing rate estimation utilities."""

import numpy as np


def get_average_cluster_firing_rate(
        spike_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        dt_ms: float = .5,
        kernel_param: float = 20.,
        kernel_type: str = "gaussian",
        n_excitatory_clusters: int | None = None,):
    """Compute average firing rates per cluster.

    Args:
        spike_matrix: Spike trains with shape `(T, n_neurons)`.
        cluster_labels: Cluster labels per neuron.
        dt_ms: Time step in milliseconds.
        kernel_param: Kernel width parameter.
        kernel_type: `"gaussian"` or `"box"`.
        n_excitatory_clusters: Number of excitatory clusters to include.
            If provided, only the first n_excitatory_clusters are returned.

    Returns:
        Array of firing rates with shape `(n_clusters, T)`.
    """
    cluster_ids, start_indices = np.unique(cluster_labels, return_index=True)
    clustered_spikes = np.add.reduceat(spike_matrix, start_indices, axis=1).T
    firing_rates = get_firing_rates(clustered_spikes, dt_ms, kernel_param, kernel_type)
    cluster_counts = np.bincount(np.searchsorted(cluster_ids, cluster_labels))
    firing_rates /= cluster_counts[:, None]
    if n_excitatory_clusters is not None:
        firing_rates = firing_rates[:n_excitatory_clusters]
    return firing_rates


def get_firing_rates(spike_train: np.ndarray,
                    dt_ms: float = .5,
                    kernel_param: float = 20.,
                    kernel_type: str = "gaussian",):
    """Smooth spike trains into firing rates.

    Args:
        spike_train: Spike trains with shape `(T, n_units)`.
        dt_ms: Time step in milliseconds.
        kernel_param: Kernel width parameter.
        kernel_type: `"gaussian"` or `"box"`.

    Returns:
        Smoothed firing rate array.
    """
    dt = dt_ms * 1e-3
    window_steps = int(kernel_param / dt_ms)
    window_steps = max(1, window_steps)  # Ensure at least 1 step

    spike_rates = spike_train / dt
    if kernel_type == "gaussian":
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(spike_rates, kernel_param)
    kernel = np.ones(window_steps) / window_steps
    return np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode='same'),
        axis=1,
        arr=spike_rates,
    )
