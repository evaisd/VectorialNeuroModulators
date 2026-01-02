"""Processing logic for spiking neural network simulations."""

from neuro_mod.core.spiking_net.processing.logic.firing_rates import (
    get_firing_rates,
    get_average_cluster_firing_rate,
)
from neuro_mod.core.spiking_net.processing.logic.detection import (
    get_activity,
    smooth_cluster_activity,
)
from neuro_mod.core.spiking_net.processing.logic.attractors import (
    get_unique_attractors,
    extract_attractors,
)
from neuro_mod.core.spiking_net.processing.logic.session_helpers import (
    load_sessions,
    get_total_duration_ms,
    get_session_cluster_spike_rates,
    get_session_cluster_activity,
    get_session_attractors_data,
    get_session_lengths_steps,
    merge_attractors_data,
)

__all__ = [
    "get_firing_rates",
    "get_average_cluster_firing_rate",
    "get_activity",
    "smooth_cluster_activity",
    "get_unique_attractors",
    "extract_attractors",
    "load_sessions",
    "get_total_duration_ms",
    "get_session_cluster_spike_rates",
    "get_session_cluster_activity",
    "get_session_attractors_data",
    "get_session_lengths_steps",
    "merge_attractors_data",
]
