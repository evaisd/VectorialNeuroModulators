"""Attractor extraction and transition analysis utilities."""

import numpy as np


def get_unique_attractors(activity_matrix: np.ndarray,):
    """Find unique overlapping activity patterns across time.

    Args:
        activity_matrix: Boolean activity matrix `(n_clusters, T)`.

    Returns:
        Sorted list of tuples describing co-active cluster indices.
    """

    rows, cols = activity_matrix.shape

    blocks = (cols + 63) // 64
    padded = np.zeros((rows, blocks * 64), dtype=bool)
    padded[:, :cols] = activity_matrix
    activity_bits = np.packbits(padded.reshape(rows, -1), axis=1, bitorder='little')

    def intersect_rows(i, j):
        return activity_bits[i] & activity_bits[j]

    cache = {}

    pairs = []
    for i in range(rows):
        for j in range(i+1, rows):
            inter = intersect_rows(i, j)
            if np.any(inter):
                key = frozenset((i, j))
                cache[key] = inter
                pairs.append((i, j))

    results = set(tuple(sorted(p)) for p in pairs)
    frontier = [tuple(sorted(p)) for p in pairs]

    while frontier:
        new_frontier = []
        for group in frontier:
            inter_vec = cache[frozenset(group)]

            # Try to add rows > max(group) to avoid duplicates
            start = group[-1] + 1
            for r in range(start, rows):
                new_inter = inter_vec & activity_bits[r]
                if np.any(new_inter):
                    new_group = tuple(sorted(group + (r,)))
                    key = frozenset(new_group)

                    if key not in cache:
                        cache[key] = new_inter
                        results.add(new_group)
                        new_frontier.append(new_group)

        frontier = new_frontier

    return sorted(results)


def extract_attractors(
        activity_matrix: np.ndarray,
        minimal_time_ms: int,
        dt_ms: float = .5,
):
    """Extract attractor states and their occurrences from activity traces.

    Args:
        activity_matrix: Boolean activity matrix `(n_clusters, T)`.
        minimal_time_ms: Minimum duration for a state to count (ms).
        dt_ms: Time step in milliseconds.

    Returns:
        Mapping from attractor identity tuples to summary dictionaries.
    """
    minimal_time = minimal_time_ms // dt_ms
    changes = np.any(activity_matrix[:, 1:] != activity_matrix[:, :-1], axis=0)
    bounds = np.concatenate(([0], np.where(changes)[0] + 1, [activity_matrix.shape[1]]))
    durations = np.diff(bounds)
    valid = durations >= minimal_time

    starts = bounds[:-1][valid].tolist()
    ends = bounds[1:][valid].tolist()
    identities = [tuple(np.flatnonzero(activity_matrix[:, s]).tolist()) for s in starts]
    attractors: dict[tuple[int, ...], dict[str, object]] = {}
    idx = 0
    for start, end, identity in zip(starts, ends, identities):
        entry = attractors.setdefault(
            identity,
            {"idx": idx,
             "#": 0,
             "starts": [],
             "ends": [],
             "occurrence_durations": [],
             "total_duration": 0,
             "clusters": identity,
             },
        )
        if entry["#"] == 0:
            idx += 1
        duration_ms = (end - start) * dt_ms
        entry["#"] += 1
        entry["starts"].append(start)
        entry["ends"].append(end)
        entry["occurrence_durations"].append(duration_ms)
        entry["total_duration"] = sum(entry["occurrence_durations"])

    # indexed: dict[int, dict[str, object]] = {}
    # for idx, identity in enumerate(sorted(attractors.keys())):
    #     entry = attractors[identity]
    #     entry["idx"] = idx
    #     entry["starts"] = tuple(entry["starts"])
    #     entry["ends"] = tuple(entry["ends"])
    #     entry["occurrence_durations"] = tuple(entry["occurrence_durations"])
    #     entry["total_duration"] = sum(entry["occurrence_durations"])
    #     indexed[idx] = entry
    # attractors.update(indexed)

    return attractors


def get_transition_matrix(
        attractors_data: dict,
):
    """Compute transition probabilities between attractors.

    Args:
        attractors_data: Mapping of attractor identities to summaries.

    Returns:
        Transition probability matrix `(n_states, n_states)`.
    """
    keys = sorted(attractors_data)
    key_to_row = {k: i for i, k in enumerate(keys)}
    n = len(keys)

    times = []
    labels = []

    for k in keys:
        s = np.asarray(attractors_data[k]["starts"])
        times.append(s)
        labels.append(np.full(len(s), key_to_row[k], dtype=int))

    times = np.concatenate(times)
    labels = np.concatenate(labels)

    order = np.argsort(times)
    labels = labels[order]

    src = labels[:-1]
    dst = labels[1:]

    counts = np.zeros((n, n), dtype=float)
    np.add.at(counts, (src, dst), 1.0)

    occ = np.array([attractors_data[k]["#"] for k in keys], dtype=float)
    occ[occ == 0] = 1.0

    return counts / occ[:, None]


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
        Tuple `(counts, occurrences)` for transitions and occurrences.
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


def get_ordered_occurrences(attractors_data: dict):
    """Return time-ordered start times and identities from attractor data."""
    times = []
    labels = []
    for identity, entry in attractors_data.items():
        starts = np.asarray(entry.get("starts", []))
        if starts.size == 0:
            continue
        times.append(starts)
        labels.append(np.array([identity] * starts.size, dtype=object))
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
    """Return unique transition pairs from ordered occurrences."""
    if labels.size < 2:
        return set()
    if session_end_times:
        session_ids = np.searchsorted(session_end_times, times, side="right")
    else:
        session_ids = None
    pairs = set()
    for idx in range(labels.size - 1):
        if session_ids is not None and session_ids[idx] != session_ids[idx + 1]:
            continue
        pairs.add((labels[idx], labels[idx + 1]))
    return pairs


def get_transition_counts_from_occurrences(
        times: np.ndarray,
        labels: np.ndarray,
        key_to_row: dict,
        session_end_times: list[float] | None = None,
):
    """Return transition counts matrix from ordered occurrences."""
    n = len(key_to_row)
    counts = np.zeros((n, n), dtype=float)
    if labels.size < 2:
        return counts
    if session_end_times:
        session_ids = np.searchsorted(session_end_times, times, side="right")
    else:
        session_ids = None
    for idx in range(labels.size - 1):
        if session_ids is not None and session_ids[idx] != session_ids[idx + 1]:
            continue
        src = key_to_row[labels[idx]]
        dst = key_to_row[labels[idx + 1]]
        counts[src, dst] += 1.0
    return counts

def get_transition_matrix_session_aware(
        attractors_data: dict,
        session_attractors: list[dict],
):
    """Compute transition probabilities across multiple sessions.

    Args:
        attractors_data: Merged attractor summaries.
        session_attractors: Per-session attractor summaries.

    Returns:
        Transition probability matrix `(n_states, n_states)`.
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
