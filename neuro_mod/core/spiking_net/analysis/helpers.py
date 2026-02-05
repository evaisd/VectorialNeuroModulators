"""Helper utilities for SNN analysis."""

from __future__ import annotations

import json
from typing import Any, Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from neuro_mod.core.spiking_net.utils import get_session_end_times_s
from neuro_mod.core.spiking_net.analysis.logic import time_window


__all__ = [
    "align_transition_matrix",
    "align_transition_matrices",
    "build_canonical_attractor_identities",
    "build_canonical_labels_from_tpms",
    "get_attractor_lex_order",
    "load_from_path",
    "load_config",
    "get_attractor_indices_in_order",
    "can_use_loaded_attractors",
    "get_attractor_identities_in_order",
    "get_unique_attractor_first_start_times",
    "get_session_end_times_s",
    "can_use_loaded_attractors",
    "build_attractor_map",
    "filter_attractors_data_between",
]


def build_attractor_map(attractors_data: dict) -> dict:
    """Build mapping from attractor index to identity.

    Args:
        attractors_data: Mapping of attractor identities to summaries.

    Returns:
        Dict mapping index to identity tuple.
    """
    return {attractors_data[k]["idx"]: k for k in attractors_data.keys()}


def _identity_sort_key(identity: Any) -> tuple[int, Any]:
    if isinstance(identity, (frozenset, set)):
        return (0, tuple(sorted(identity)))
    if isinstance(identity, (tuple, list)):
        return (0, tuple(identity))
    if isinstance(identity, (int, float, str)):
        return (1, identity)
    return (1, repr(identity))


def build_canonical_attractor_identities(
    attractors_runs: Iterable[dict],
) -> list[Any]:
    identities: set[Any] = set()
    for attractors_data in attractors_runs:
        identities.update(attractors_data.keys())
    return sorted(identities, key=_identity_sort_key)


def build_canonical_labels_from_tpms(
    tpms: Iterable[pd.DataFrame],
) -> list[Any]:
    labels: set[Any] = set()
    for tpm in tpms:
        labels.update(tpm.index)
        labels.update(tpm.columns)
    return sorted(labels, key=_identity_sort_key)


def align_transition_matrix(
    tpm: pd.DataFrame,
    canonical_labels: Sequence[Any],
) -> pd.DataFrame:
    return tpm.reindex(index=canonical_labels, columns=canonical_labels, fill_value=0.0)


def align_transition_matrices(
    tpms: dict[str, pd.DataFrame],
    *,
    canonical_labels: Sequence[Any] | None = None,
) -> tuple[list[Any], dict[str, pd.DataFrame]]:
    if canonical_labels is None:
        canonical_labels = build_canonical_labels_from_tpms(tpms.values())
    aligned = {
        key: align_transition_matrix(tpm, canonical_labels)
        for key, tpm in tpms.items()
    }
    return list(canonical_labels), aligned


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


def load_from_path(path: Path) -> dict:
    """Load attractors_data from a directory.

    Args:
        path: Directory containing saved processed data.

    Returns:
        The attractors_data dictionary.
    """
    config_path = path / "processor_config.json"

    if config_path.exists():
        config = json.loads(config_path.read_text())
        files = config.get("files", {})
        attractors_filename = files.get("attractors", "attractors.npy")
    else:
        attractors_filename = "attractors.npy"

    attractors_path = path / attractors_filename
    data = np.load(attractors_path, allow_pickle=True).item()

    return data


def load_config(path: Path) -> dict:
    """Load configuration from a directory.

    Args:
        path: Directory containing saved config.

    Returns:
        The configuration dictionary.
    """
    config_path = path / "processor_config.json"
    batch_config_path = path / "batch_config.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text())
    if batch_config_path.exists():
        batch_config = json.loads(batch_config_path.read_text())
        config.setdefault("batch", batch_config)
        repeats = batch_config.get("repeats", [])
        repeat_durations = [
            repeat.get("duration_ms")
            for repeat in repeats
            if repeat.get("duration_ms") is not None
        ]
        config.setdefault("repeat_durations_ms", repeat_durations)
        config.setdefault("n_runs", batch_config.get("n_runs"))
    return config


def get_attractor_indices_in_order(
    attractors_data: dict,
) -> list[Any]:
    if not attractors_data:
        return []
    identities = [
        identity
        for identity, entry in sorted(
            attractors_data.items(),
            key=lambda item: item[1].get("idx", 0),
        )
    ]
    return [attractors_data[identity].get("idx", identity) for identity in identities]


def get_attractor_lex_order(
    attractors_data: dict,
) -> tuple[list[Any], list[Any]]:
    """Return lex-ordered attractor indices and identities.

    Ordering is based on the attractor identity (cluster pattern) using the
    same sort key as canonical label building.
    """
    if not attractors_data:
        return [], []
    mapping = build_attractor_map(attractors_data)
    ordered = sorted(mapping.items(), key=lambda item: _identity_sort_key(item[1]))
    ordered_indices = [idx for idx, _ in ordered]
    ordered_identities = [identity for _, identity in ordered]
    return ordered_indices, ordered_identities
