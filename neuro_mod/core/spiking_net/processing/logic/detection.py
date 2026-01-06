"""Activity detection utilities for spiking network outputs."""

import numpy as np


def get_activity(
        firing_rates: np.ndarray,
        baseline_rate: float | np.ndarray = None,
        flag: bool = False,
        *,
        max_iters: int = 50,
        tol: float = 20.0,
):
    """Determine active clusters based on firing rates.

    Args:
        firing_rates: Array of firing rates per cluster over time.
        baseline_rate: Optional baseline rate for comparison.
        flag: If True, return after the first pass (compatibility).
        max_iters: Maximum refinement iterations before returning.
        tol: Convergence threshold for baseline updates.

    Returns:
        Boolean activity matrix of the same shape as `firing_rates`.
    """
    if baseline_rate is None:
        baseline_vec = firing_rates.mean(axis=1)
    else:
        baseline_arr = np.asarray(baseline_rate, dtype=float)
        if baseline_arr.ndim == 0:
            baseline_vec = np.full(firing_rates.shape[0], float(baseline_arr))
        else:
            baseline_vec = baseline_arr

    active_matrix = firing_rates > baseline_vec[:, None]
    if flag:
        return active_matrix
    for _ in range(max_iters):
        if not np.any(active_matrix):
            return active_matrix
        updated_baseline = firing_rates[active_matrix].mean()
        if not np.isfinite(updated_baseline):
            return active_matrix
        if abs(updated_baseline - baseline_vec.mean()) < tol:
            return active_matrix
        baseline_vec = np.full(firing_rates.shape[0], float(updated_baseline))
        active_matrix = firing_rates > baseline_vec[:, None]

    return active_matrix


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

    for row_idx in range(activity_matrix.shape[0]):
        row = activity_matrix[row_idx]
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
                out[row_idx, gap_start:gap_end] = True

    return out
