
import numpy as np


def get_average_cluster_firing_rate(
        spike_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        dt_ms: float = .5,
        kernel_param: float = 20.,
        kernel_type: str = "gaussian",):
    unique_clusters, indices = np.unique(cluster_labels, return_index=True)
    # firing_rates = get_firing_rates(spike_matrix, dt_ms, kernel_param, kernel_type).T
    clustered_spikes = np.add.reduceat(spike_matrix, indices, axis=1).T
    firing_rates = get_firing_rates(clustered_spikes, dt_ms, kernel_param, kernel_type)
    firing_rates /= np.bincount(np.searchsorted(unique_clusters, cluster_labels))[:, None]
    return firing_rates


def get_firing_rates(spike_train: np.ndarray,
                    dt_ms: float = .5,
                    kernel_param: float = 20.,
                    kernel_type: str = "gaussian",):
    dt = dt_ms * 1e-3
    window_steps = int(kernel_param / dt_ms)
    window_steps = max(1, window_steps)  # Ensure at least 1 step

    if kernel_type == "gaussian":
        from scipy.ndimage import gaussian_filter1d
        smoothed_spikes = gaussian_filter1d(spike_train / dt, kernel_param)
    else:
        kernel = np.ones(window_steps) / window_steps
        smoothed_spikes = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode='same'),
            axis=1,
            arr=spike_train / dt
        )
    return smoothed_spikes
