"""Experiment pipeline for neuroscience simulations.

This module provides a generic Pipeline class that orchestrates the full
experiment workflow: Simulation -> Processing -> Analysis -> Plotting.

The Pipeline handles all execution modes internally:
- SINGLE: Single simulation run
- REPEATED: Multiple runs with seed management
- SWEEP: Parameter variations
- SWEEP_REPEATED: Factorial design (sweep × repeats)

Example:
    >>> from neuro_mod.pipeline import Pipeline, PipelineConfig, ExecutionMode
    >>> from neuro_mod.execution import StageSNNSimulation
    >>> from neuro_mod.core.spiking_net.processing import SNNProcessor
    >>> from neuro_mod.core.spiking_net.analysis import SNNAnalyzer
    >>>
    >>> pipeline = Pipeline(
    ...     simulator_factory=lambda seed: StageSNNSimulation(config, random_seed=seed),
    ...     processor_factory=lambda raw: SNNProcessor(raw["spikes_path"]),
    ...     analyzer_factory=SNNAnalyzer,
    ... )
    >>>
    >>> result = pipeline.run(PipelineConfig(
    ...     mode=ExecutionMode.REPEATED,
    ...     n_repeats=10,
    ...     parallel=True,
    ...     save_dir=Path("experiments/run_001"),
    ... ))
    >>>
    >>> df = result.dataframes["aggregated"]
    >>> metrics = result.metrics["aggregated"]
"""

from neuro_mod.pipeline.config import ExecutionMode, PipelineConfig, PipelineResult
from neuro_mod.pipeline.pipeline import Pipeline
from neuro_mod.pipeline.plotting import (
    BasePlotter,
    ComposablePlotter,
    MatplotlibPlotter,
    PlotSpec,
    SpecPlotter,
    SeabornPlotter,
)
from neuro_mod.analysis.base_analyzer import BaseAnalyzer, MetricResult, manipulation, metric
from neuro_mod.pipeline.protocols import (
    Analyzer,
    AnalyzerFactory,
    Plotter,
    Processor,
    ProcessorFactory,
    Simulator,
    SimulatorFactory,
)

__all__ = [
    # Core Pipeline
    "Pipeline",
    "PipelineConfig",
    "PipelineResult",
    "ExecutionMode",
    # Protocols
    "Simulator",
    "SimulatorFactory",
    "Processor",
    "ProcessorFactory",
    "Analyzer",
    "AnalyzerFactory",
    "Plotter",
    # Plotters
    "BasePlotter",
    "SpecPlotter",
    "SeabornPlotter",
    "PlotSpec",
    "ComposablePlotter",
    "MatplotlibPlotter",
    # Analyzer base
    "BaseAnalyzer",
    "MetricResult",
    "manipulation",
    "metric",
]
