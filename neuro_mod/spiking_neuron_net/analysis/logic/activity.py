
import numpy as np


def get_cluster_activity(
        spike_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        dt_ms: float = 1.,
        kernel_param: float = 20.,
        kernel_type: str = "uniform",
        baseline_rate: float = None,
):
    firing_rates = get_average_cluster_firing_rate(
        spike_matrix, cluster_labels, dt_ms, kernel_param, kernel_type
    )
    baseline_rate = firing_rates[:].mean(axis=(1,)) if baseline_rate is None else baseline_rate
    active_matrix = firing_rates > baseline_rate[:, None]
    return firing_rates, active_matrix


def get_average_cluster_firing_rate(
        spike_matrix: np.ndarray,
        cluster_labels: np.ndarray,
        dt_ms: float = 1.,
        kernel_param: float = 20.,
        kernel_type: str = "uniform",):
    unique_clusters, indices = np.unique(cluster_labels, return_index=True)
    firing_rates = get_firing_rates(spike_matrix, dt_ms, kernel_param, kernel_type)
    firing_rates = np.add.reduceat(firing_rates, indices, axis=0)
    # firing_rates = get_firing_rates(clustered_spikes, dt_ms, kernel_param, kernel_type)
    firing_rates /= np.bincount(np.searchsorted(unique_clusters, cluster_labels))[:, None]
    return firing_rates


def get_firing_rates(spike_train: np.ndarray,
                    dt_ms: float = 1.,
                    kernel_param: float = 20.,
                    kernel_type: str = "uniform",):
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


def detect_active_clusters_distributed(
        cluster_rate: np.ndarray,
        threshold: float | np.ndarray = None,
        threshold_z: float = 3,
        dt: float = 1e-3,
        min_duration_ms: float = 50.) -> np.ndarray:
    cluster_rate = np.asarray(cluster_rate)
    if cluster_rate.ndim != 2:
        raise ValueError("cluster_rate must have shape (C, T)")

    C, T = cluster_rate.shape
    min_steps = max(1, int(round(min_duration_ms / 1000.0 / dt)))

    mean_rates = np.median(cluster_rate, axis=1, keepdims=True)
    std_rates = cluster_rate.std(axis=1, keepdims=True)
    threshold = np.maximum(mean_rates + threshold_z * std_rates, threshold)

    if np.isscalar(threshold):
        threshold = np.array(threshold)
    if threshold.ndim == 1:
        threshold = threshold[:, None]

    above = cluster_rate > threshold
    padded = np.pad(above.astype(int), ((0, 0), (1, 1)), mode='constant')
    diff = np.diff(padded, axis=1)

    start_cs, start_ts = np.where(diff == 1)
    end_cs, end_ts = np.where(diff == -1)

    durations = end_ts - start_ts
    valid_mask = durations >= min_steps

    valid_starts_c = start_cs[valid_mask]
    valid_starts_t = start_ts[valid_mask]
    valid_ends_c = end_cs[valid_mask]
    valid_ends_t = end_ts[valid_mask]

    flat_starts = valid_starts_c * (T + 1) + valid_starts_t
    flat_ends = valid_ends_c * (T + 1) + valid_ends_t

    indicator = np.zeros(C * (T + 1), dtype=int)
    np.add.at(indicator, flat_starts, 1)
    np.add.at(indicator, flat_ends, -1)

    indicator = indicator.reshape((C, T + 1))
    active_matrix = indicator.cumsum(axis=1)[:, :-1].astype(bool)

    return active_matrix


def get_activity_stats(
        firing_rates: np.ndarray,
        active_matrix: np.ndarray,
):
    active_clusters_t = active_matrix.sum(axis=0)
    prob_n_active = np.bincount(active_clusters_t) / active_clusters_t.size
    most_likely = np.unique(active_clusters_t)[prob_n_active.argmax()]
    return most_likely, prob_n_active, active_clusters_t