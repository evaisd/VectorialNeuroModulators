"""Time window helpers for analysis."""

from __future__ import annotations

import warnings


def resolve_time_bounds_s(
        total_duration_s: float,
        t_from: float | None,
        t_to: float | None,
) -> tuple[float, float]:
    """Resolve time bounds in seconds with negative offsets and warnings."""
    t_from_val = 0.0 if t_from is None else t_from
    t_to_val = total_duration_s if t_to is None else t_to
    for t in (t_from_val, t_to_val):
        if abs(t) > total_duration_s:
            warnings.warn(
                "Absolute time threshold exceeds total duration; returning full attractor data.",
                RuntimeWarning,
            )
            return 0.0, total_duration_s
    if t_from_val < 0:
        t_from_s = total_duration_s + t_from_val
    else:
        t_from_s = t_from_val
    if t_to_val < 0:
        t_to_s = total_duration_s + t_to_val
    else:
        t_to_s = t_to_val
    if t_from_s > t_to_s:
        warnings.warn(
            "t_from is greater than t_to; swapping bounds.",
            RuntimeWarning,
        )
        t_from_s, t_to_s = t_to_s, t_from_s
    return t_from_s, t_to_s


def get_time_bounds_steps(
        total_duration_s: float,
        dt_s: float,
        t_from: float | None,
        t_to: float | None,
) -> tuple[int, int]:
    """Return start/end bounds in steps for the provided time window."""
    t_from_s, t_to_s = resolve_time_bounds_s(total_duration_s, t_from, t_to)
    return int(t_from_s // dt_s), int(t_to_s // dt_s)
