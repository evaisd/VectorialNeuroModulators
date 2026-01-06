"""Session handling utilities for SNN data processing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neuro_mod.core.spiking_net.processing.logic import firing_rates as fr
from neuro_mod.core.spiking_net.processing.logic import detection
from neuro_mod.core.spiking_net.processing.logic import attractors
from neuro_mod.core.spiking_net.utils import get_session_end_times_s


def iter_spike_files(spikes_path: Path):
    """Iterate over spike files in a path.

    Args:
        spikes_path: Path to a single file or directory of files.

    Yields:
        Paths to spike files.
    """
    if spikes_path.is_file():
        yield spikes_path
    else:
        for file_path in sorted(spikes_path.glob("*.np[yz]")):
            yield file_path


def read_spikes_clusters(spikes_path: Path, clusters_path: Path | None):
    """Read spike and cluster data from files.

    Args:
        spikes_path: Path to spike data file.
        clusters_path: Optional path to cluster labels file.

    Returns:
        Tuple of (spikes, clusters) arrays.
    """
    data = np.load(spikes_path, allow_pickle=True)
    if spikes_path.suffix == ".npz":
        spikes = data["spikes"]
    else:
        spikes = data
    if clusters_path is None:
        clusters = data["clusters"]
    else:
        clusters = np.load(clusters_path)
    return spikes, clusters


def load_sessions(spikes_path: Path, clusters_path: Path | None):
    """Load all sessions from spike files.

    Args:
        spikes_path: Path to spike data file or directory.
        clusters_path: Optional path to cluster labels file.

    Returns:
        Tuple of (spikes, clusters) tuples for each session.
    """
    sessions = []
    clusters_ref = None
    for spikes_file in iter_spike_files(spikes_path):
        spikes, clusters = read_spikes_clusters(spikes_file, clusters_path)
        if clusters_ref is None:
            clusters_ref = clusters
        elif clusters.shape != clusters_ref.shape or not np.array_equal(clusters, clusters_ref):
            raise ValueError("Cluster labels differ across simulations.")
        sessions.append((spikes, clusters))
    return tuple(sessions)


def get_total_duration_ms(sessions, dt: float) -> float:
    """Calculate total duration across all sessions.

    Args:
        sessions: List of (spikes, clusters) tuples.
        dt: Time step in seconds.

    Returns:
        Total duration in milliseconds.
    """
    return dt * sum(spikes.shape[0] for spikes, _ in sessions) * 1e3


def aggregate_series(series, axis: int):
    """Concatenate a list of arrays along an axis.

    Args:
        series: List of arrays.
        axis: Axis to concatenate along.

    Returns:
        Concatenated array, or empty array if list is empty.
    """
    if not series:
        return np.empty(())
    if len(series) == 1:
        return series[0]
    return np.concatenate(series, axis=axis)


def get_session_cluster_spike_rates(
        sessions,
        clustering_params: dict,
        dt: float,
) -> list[np.ndarray]:
    """Compute cluster firing rates for each session.

    Args:
        sessions: List of (spikes, clusters) tuples.
        clustering_params: Parameters for firing rate computation.
        dt: Time step in seconds.

    Returns:
        List of firing rate arrays per session.
    """
    params = clustering_params.copy()
    params.setdefault("dt_ms", dt / 1e-3)
    rates = []
    for spikes, clusters in sessions:
        rates.append(
            fr.get_average_cluster_firing_rate(
                spikes,
                clusters,
                **params
            )
        )
    return rates


def get_session_cluster_activity(session_cluster_rates: list[np.ndarray]) -> list[np.ndarray]:
    """Compute binary activity matrices for each session.

    Args:
        session_cluster_rates: List of firing rate arrays per session.

    Returns:
        List of boolean activity arrays per session.
    """
    return [detection.get_activity(cluster_rates) for cluster_rates in session_cluster_rates]


def get_session_attractors_data(
        session_cluster_activity: list[np.ndarray],
        minimal_time_ms: float,
        dt_ms: float,
) -> list[dict]:
    """Extract attractors for each session.

    Args:
        session_cluster_activity: List of activity matrices per session.
        minimal_time_ms: Minimum attractor duration in milliseconds.
        dt_ms: Time step in milliseconds.

    Returns:
        List of attractor dictionaries per session.
    """
    session_attractors = []
    for activity_matrix in session_cluster_activity:
        session_attractors.append(
            attractors.extract_attractors(activity_matrix, minimal_time_ms, dt_ms)
        )
    return session_attractors


def get_session_lengths_steps(session_cluster_activity: list[np.ndarray]) -> list[int]:
    """Get the length of each session in time steps.

    Args:
        session_cluster_activity: List of activity matrices per session.

    Returns:
        List of session lengths in time steps.
    """
    return [mat.shape[1] for mat in session_cluster_activity]


def validate_no_cross_simulation_attractors(session_attractors, session_lengths):
    """Validate that no attractors cross session boundaries.

    Args:
        session_attractors: List of attractor dicts per session.
        session_lengths: List of session lengths in time steps.

    Raises:
        ValueError: If attractors cross session boundaries.
    """
    if len(session_attractors) != len(session_lengths):
        raise ValueError("Session lengths do not match attractor sessions.")
    offsets = np.cumsum([0] + session_lengths[:-1]).tolist()
    for session_idx, attractors_data in enumerate(session_attractors):
        session_len = session_lengths[session_idx]
        offset = offsets[session_idx]
        for identity, entry in attractors_data.items():
            starts = entry.get("starts", [])
            ends = entry.get("ends", [])
            if len(starts) != len(ends):
                raise ValueError(f"Mismatched starts/ends for attractor {identity}.")
            prev_end = None
            for start, end in zip(starts, ends):
                if start < 0 or end > session_len or end <= start:
                    raise ValueError(
                        f"Invalid start/end for attractor {identity} in session {session_idx}."
                    )
                global_start = start + offset
                global_end = end + offset
                if global_start < offset or global_end > offset + session_len:
                    raise ValueError(
                        f"Attractor {identity} crosses simulation boundary in session {session_idx}."
                    )
                if prev_end is not None and start < prev_end:
                    raise ValueError(
                        f"Attractor {identity} has overlapping occurrences in session {session_idx}."
                    )
                prev_end = end


def merge_attractors_data(session_attractors, session_lengths, dt: float):
    """Merge attractors across sessions with time offset adjustment.

    Args:
        session_attractors: List of attractor dicts per session.
        session_lengths: List of session lengths in time steps.
        dt: Time step in seconds.

    Returns:
        Merged attractor dictionary with starts/ends in seconds.
    """
    merged = {}
    next_idx = 0
    offsets_s = np.cumsum(
        [0.0] + [round(length * dt, 4) for length in session_lengths[:-1]]
    ).tolist()
    for session_idx, attractors_data in enumerate(session_attractors):
        offset = offsets_s[session_idx]
        for identity, entry in attractors_data.items():
            if identity not in merged:
                merged[identity] = {
                    "idx": next_idx,
                    "#": 0,
                    "starts": [],
                    "ends": [],
                    "occurrence_durations": [],
                    "total_duration": 0,
                    "clusters": identity,
                }
                next_idx += 1
            merged_entry = merged[identity]
            merged_entry["#"] += entry["#"]
            merged_entry["starts"].extend(
                [round((s * dt) + offset, 4) for s in entry["starts"]]
            )
            merged_entry["ends"].extend(
                [round((e * dt) + offset, 4) for e in entry["ends"]]
            )
            merged_entry["occurrence_durations"].extend(entry["occurrence_durations"])
            merged_entry["total_duration"] += entry["total_duration"]
    for entry in merged.values():
        starts = entry["starts"]
        if len(starts) <= 1:
            continue
        order = sorted(range(len(starts)), key=starts.__getitem__)
        entry["starts"] = [starts[i] for i in order]
        entry["ends"] = [entry["ends"][i] for i in order]
        entry["occurrence_durations"] = [entry["occurrence_durations"][i] for i in order]
    return merged


__all__ = [
    "aggregate_series",
    "get_session_attractors_data",
    "get_session_cluster_activity",
    "get_session_cluster_spike_rates",
    "get_session_end_times_s",
    "get_session_lengths_steps",
    "get_total_duration_ms",
    "iter_spike_files",
    "load_sessions",
    "merge_attractors_data",
    "read_spikes_clusters",
    "validate_no_cross_simulation_attractors",
]
