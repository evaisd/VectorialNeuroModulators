"""Shared utilities for spiking network modules."""

from __future__ import annotations

import numpy as np


def get_session_end_times_s(session_lengths_steps: list[int], dt: float) -> list[float]:
    """Get session end times in seconds from step lengths."""
    return (np.cumsum(session_lengths_steps) * dt).tolist()
