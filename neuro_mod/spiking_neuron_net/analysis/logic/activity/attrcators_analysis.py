
import numpy as np


def get_unique_attractors(activity_matrix: np.ndarray,):

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

