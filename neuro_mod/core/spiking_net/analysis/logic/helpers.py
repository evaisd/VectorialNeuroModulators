"""Helper utilities for SNN analysis."""

from __future__ import annotations

import numpy as np

from neuro_mod.core.spiking_net.utils import get_session_end_times_s

from neuro_mod.core.spiking_net.analysis.logic import time_window


def build_attractor_map(attractors_data: dict) -> dict:
    """Build mapping from attractor index to identity.

    Args:
        attractors_data: Mapping of attractor identities to summaries.

    Returns:
        Dict mapping index to identity tuple.
    """
    return {attractors_data[k]["idx"]: k for k in attractors_data.keys()}


def convert_attractors_data_steps_to_seconds(attractors_data: dict, dt: float) -> dict:
    """Convert attractor start/end times from steps to seconds.

    Args:
        attractors_data: Attractor data with times in steps.
        dt: Time step in seconds.

    Returns:
        Attractor data with times in seconds.
    """
    for entry in attractors_data.values():
        starts = entry.get("starts", [])
        ends = entry.get("ends", [])
        entry["starts"] = [round(float(s) * dt, 4) for s in starts]
        entry["ends"] = [round(float(e) * dt, 4) for e in ends]
    return attractors_data


def filter_attractors_data_between(
        attractors_data: dict,
        total_duration_s: float,
        t_from: float | None,
        t_to: float | None,
) -> dict:
    """Filter attractor data to a time window.

    Args:
        attractors_data: Full attractor data.
        total_duration_s: Total simulation duration in seconds.
        t_from: Start of time window (seconds).
        t_to: End of time window (seconds).

    Returns:
        Filtered attractor data containing only occurrences in window.
    """
    t_from_s, t_to_s = time_window.resolve_time_bounds_s(
        total_duration_s,
        t_from,
        t_to,
    )
    filtered = {}
    for identity, entry in attractors_data.items():
        starts = entry.get("starts", [])
        if not starts:
            continue
        keep_indices = [i for i, s in enumerate(starts) if t_from_s <= s <= t_to_s]
        if not keep_indices:
            continue
        ends = entry.get("ends", [])
        durations = entry.get("occurrence_durations", [])
        filtered_entry = {
            "idx": entry.get("idx"),
            "#": len(keep_indices),
            "starts": [starts[i] for i in keep_indices],
            "ends": [ends[i] for i in keep_indices],
            "occurrence_durations": [durations[i] for i in keep_indices],
            "total_duration": float(np.sum([durations[i] for i in keep_indices])),
            "clusters": entry.get("clusters", identity),
        }
        filtered[identity] = filtered_entry
    return filtered


def get_attractor_identities_in_order(attractors_data: dict) -> list[tuple[int, ...]]:
    """Get attractor identities sorted by their index.

    Args:
        attractors_data: Mapping of attractor identities to summaries.

    Returns:
        List of identity tuples in index order.
    """
    return [
        identity
        for identity, entry in sorted(
            attractors_data.items(),
            key=lambda item: item[1].get("idx", 0),
        )
    ]


def get_unique_attractor_first_start_times(attractors_data: dict) -> np.ndarray:
    """Get the first start time for each unique attractor.

    Args:
        attractors_data: Mapping of attractor identities to summaries.

    Returns:
        Array of first start times.
    """
    first_starts = []
    for entry in attractors_data.values():
        starts = entry.get("starts", [])
        if not starts:
            continue
        first_starts.append(min(starts))
    if not first_starts:
        return np.empty((0,), dtype=float)
    return np.asarray(first_starts, dtype=float)


def can_use_loaded_attractors(
        has_attractors_data: bool,
        kwargs: dict,
        minimal_life_span_ms: float,
) -> bool:
    """Check if loaded attractors data can be reused.

    Args:
        has_attractors_data: Whether attractors_data exists.
        kwargs: Additional parameters passed.
        minimal_life_span_ms: Default minimal lifespan.

    Returns:
        True if loaded data can be used without reprocessing.
    """
    if not has_attractors_data:
        return False
    if not kwargs:
        return True
    if set(kwargs.keys()) == {"minimal_time_ms"}:
        return kwargs["minimal_time_ms"] == minimal_life_span_ms
    return False
