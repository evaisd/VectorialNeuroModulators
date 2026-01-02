"""Analysis tools for spiking neuron network simulations."""

from neuro_mod.core.spiking_net.analysis.snn_analyzer import SNNAnalyzer
from neuro_mod.core.spiking_net.analysis import plotting
from neuro_mod.core.spiking_net.analysis import logic

# Legacy import - will be removed after migration
from neuro_mod.core.spiking_net.analysis.analyzer import Analyzer

__all__ = [
    "SNNAnalyzer",
    "Analyzer",  # Legacy
    "plotting",
    "logic",
]
