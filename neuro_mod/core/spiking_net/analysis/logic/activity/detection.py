"""Activity detection utilities for spiking network outputs."""

import numpy as np


def get_activity(
        firing_rates: np.ndarray,
        baseline_rate: float | np.ndarray = None,
        flag: bool = False,
):
    """Determine active clusters based on firing rates.

    Args:
        firing_rates: Array of firing rates per cluster over time.
        baseline_rate: Optional baseline rate for comparison.
        flag: Internal recursion flag.

    Returns:
        Boolean activity matrix of the same shape as `firing_rates`.
    """
    if isinstance(baseline_rate, float):
        baseline_rate = np.full(firing_rates.shape[0], baseline_rate)
    baseline_rate = firing_rates[:].mean(axis=(1,)) if baseline_rate is None else baseline_rate
    active_matrix = firing_rates > baseline_rate[:, None]
    if flag:
        return active_matrix
    _baseline = firing_rates[active_matrix].mean()
    if abs(_baseline - baseline_rate.mean()) < 20.:
        return get_activity(firing_rates, _baseline, True)
    return get_activity(firing_rates, _baseline, flag)


def smooth_cluster_activity(
        activity_matrix: np.ndarray,
        minimal_length_ms: float = 100.,
        dt_ms: float = .5
):
    """Fill short gaps in binary activity traces.

    Args:
        activity_matrix: Boolean activity array `(n_clusters, T)`.
        minimal_length_ms: Minimum gap length to keep (ms).
        dt_ms: Time step in milliseconds.

    Returns:
        Smoothed activity matrix.
    """
    out = activity_matrix.copy()
    minimal_length = minimal_length_ms // dt_ms

    for r in range(activity_matrix.shape[0]):
        row = activity_matrix[r]
        if not np.any(row):
            continue

        padded = np.pad(row, (1, 1), constant_values=False)
        diff = np.diff(padded.astype(int))
        true_starts = np.where(diff == 1)[0]
        true_ends = np.where(diff == -1)[0]

        for i in range(len(true_starts) - 1):
            gap_start = true_ends[i]
            gap_end = true_starts[i + 1]
            gap_len = gap_end - gap_start
            if 0 < gap_len < minimal_length:
                out[r, gap_start:gap_end] = True

    return out
