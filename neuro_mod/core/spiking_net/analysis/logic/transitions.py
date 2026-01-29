"""Transition matrix computation utilities for attractor analysis."""

from __future__ import annotations

import numpy as np


def get_ordered_occurrences(attractors_data: dict):
    """Return time-ordered start times and integer indices from attractor data.

    Args:
        attractors_data: Mapping of attractor identities to summaries.

    Returns:
        Tuple of (times, labels) arrays sorted by start time.
    """
    times = []
    labels = []
    for identity, entry in attractors_data.items():
        starts = np.asarray(entry.get("starts", [])).ravel()
        if starts.size == 0:
            continue
        times.append(starts)
        idx = entry.get("idx")
        if idx is None:
            if np.isscalar(identity) or isinstance(identity, (str, bytes)):
                idx = identity
            else:
                idx = tuple(np.asarray(identity).ravel().tolist())
        labels.append(np.full(starts.size, idx, dtype=object))
    if not times:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=object)
    times = np.concatenate(times)
    labels = np.concatenate(labels)
    order = np.argsort(times)
    return times[order], labels[order]


def get_transition_pairs(
        times: np.ndarray,
        labels: np.ndarray,
        session_end_times: list[float] | None = None,
):
    """Return unique transition pairs from ordered occurrences.

    Args:
        times: Array of start times.
        labels: Array of attractor labels.
        session_end_times: Optional list of session boundaries.

    Returns:
        Set of (source, destination) label pairs.
    """
    if labels.size < 2:
        return set()
    session_ids = _get_session_ids(times, session_end_times)
    pairs = set()
    for step_idx in range(labels.size - 1):
        if session_ids is not None and session_ids[step_idx] != session_ids[step_idx + 1]:
            continue
        pairs.add((labels[step_idx], labels[step_idx + 1]))
    return pairs


def get_transition_counts(
        attractors_data: dict,
        key_to_row: dict,
        n: int,
):
    """Compute transition counts and occurrences for a session.

    Args:
        attractors_data: Attractor summaries for a session.
        key_to_row: Mapping from attractor identities to row indices.
        n: Number of unique attractors.

    Returns:
        Tuple (counts, occurrences) for transitions and occurrences.
    """
    times = []
    labels = []
    occ = np.zeros(n, dtype=float)
    for identity, row in key_to_row.items():
        entry = attractors_data.get(identity)
        if entry is None:
            continue
        occ[row] = entry["#"]
        starts = np.asarray(entry["starts"])
        if starts.size == 0:
            continue
        times.append(starts)
        labels.append(np.full(starts.size, row, dtype=int))
    counts = np.zeros((n, n), dtype=float)
    if times:
        times = np.concatenate(times)
        labels = np.concatenate(labels)
        order = np.argsort(times)
        labels = labels[order]
        if labels.size > 1:
            src = labels[:-1]
            dst = labels[1:]
            np.add.at(counts, (src, dst), 1.0)
    return counts, occ


def get_transition_counts_from_occurrences(
        times: np.ndarray,
        labels: np.ndarray,
        key_to_row: dict,
        session_end_times: list[float] | None = None,
):
    """Return transition counts matrix from ordered occurrences.

    Args:
        times: Array of start times.
        labels: Array of attractor labels.
        key_to_row: Mapping from labels to row indices.
        session_end_times: Optional list of session boundaries.

    Returns:
        Transition counts matrix.
    """
    n = len(key_to_row)
    counts = np.zeros((n, n), dtype=float)
    if labels.size < 2:
        return counts
    session_ids = _get_session_ids(times, session_end_times)
    for step_idx in range(labels.size - 1):
        if session_ids is not None and session_ids[step_idx] != session_ids[step_idx + 1]:
            continue
        src = key_to_row[labels[step_idx]]
        dst = key_to_row[labels[step_idx + 1]]
        counts[src, dst] += 1.0
    return counts


def get_transition_matrix(attractors_data: dict) -> np.ndarray:
    """Compute transition probabilities between attractors.

    Args:
        attractors_data: Mapping of attractor identities to summaries.

    Returns:
        Transition probability matrix (n_states, n_states).
    """
    if not attractors_data:
        return np.zeros((0, 0), dtype=float)
    keys = sorted(attractors_data)
    key_to_row = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    counts, occ = get_transition_counts(attractors_data, key_to_row, n)
    occ[occ == 0] = 1.0
    return counts / occ[:, None]


def get_transition_matrix_from_data(
        attractors_data: dict,
        session_end_times: list[float] | None = None,
) -> np.ndarray:
    """Compute transition probabilities from attractor data.

    Args:
        attractors_data: Mapping of attractor identities to summaries.
        session_end_times: Optional list of session boundaries.
        filter_by_size: filter attractors of certain sizes.


    Returns:
        Transition probability matrix (n_states, n_states).
    """
    if not attractors_data:
        return np.zeros((0, 0), dtype=float)
    identities = [
        identity
        for identity, entry in sorted(
            attractors_data.items(),
            key=lambda item: item[1].get("idx", 0),
        )
    ]
    key_to_row = {k: i for i, k in enumerate(identities)}
    idx_to_row = {attractors_data[k].get("idx", k): key_to_row[k] for k in identities}
    n = len(identities)
    times, labels = get_ordered_occurrences(attractors_data)
    if times.size == 0 or labels.size < 2:
        return np.zeros((n, n), dtype=float)
    occ = np.array([attractors_data[k].get("#", 0) for k in identities], dtype=float)
    counts = get_transition_counts_from_occurrences(
        times,
        labels,
        idx_to_row,
        session_end_times=session_end_times,
    )
    occ[occ == 0] = 1.0
    return counts / occ[:, None]


def get_transition_matrix_session_aware(
        attractors_data: dict,
        session_attractors: list[dict],
) -> np.ndarray:
    """Compute transition probabilities across multiple sessions.

    Args:
        attractors_data: Merged attractor summaries.
        session_attractors: Per-session attractor summaries.

    Returns:
        Transition probability matrix (n_states, n_states).
    """
    if not attractors_data:
        return np.zeros((0, 0), dtype=float)
    keys = sorted(attractors_data)
    key_to_row = {k: i for i, k in enumerate(keys)}
    n = len(keys)
    total_counts = np.zeros((n, n), dtype=float)
    total_occ = np.zeros(n, dtype=float)
    for session_data in session_attractors:
        counts, occ = get_transition_counts(session_data, key_to_row, n)
        total_counts += counts
        total_occ += occ
    total_occ[total_occ == 0] = 1.0
    return total_counts / total_occ[:, None]


def _get_session_ids(
        times: np.ndarray,
        session_end_times: list[float] | None,
):
    """Get session IDs for each time point.

    Args:
        times: Array of time points.
        session_end_times: List of session end times.

    Returns:
        Array of session IDs, or None if no session boundaries.
    """
    if session_end_times:
        return np.searchsorted(session_end_times, times, side="right")
    return None
