"""Attractor extraction utilities for processing spiking network outputs."""

import numpy as np


def get_unique_attractors(activity_matrix: np.ndarray,):
    """Find unique overlapping activity patterns across time.

    Args:
        activity_matrix: Boolean activity matrix `(n_clusters, T)`.

    Returns:
        Sorted list of tuples describing co-active cluster indices.
    """

    n_rows, n_cols = activity_matrix.shape

    blocks = (n_cols + 63) // 64
    padded = np.zeros((n_rows, blocks * 64), dtype=bool)
    padded[:, :n_cols] = activity_matrix
    activity_bits = np.packbits(padded.reshape(n_rows, -1), axis=1, bitorder='little')

    def intersect_rows(i, j):
        return activity_bits[i] & activity_bits[j]

    cache = {}

    pairs = []
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
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
            for r in range(start, n_rows):
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
        minimal_time_ms: float,
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
    minimal_steps = int(minimal_time_ms // dt_ms)
    changes = np.any(activity_matrix[:, 1:] != activity_matrix[:, :-1], axis=0)
    bounds = np.concatenate(([0], np.where(changes)[0] + 1, [activity_matrix.shape[1]]))
    durations = np.diff(bounds)
    valid = durations >= minimal_steps

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
        entry["total_duration"] += duration_ms

    return attractors
