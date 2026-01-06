# Pipeline Module

A generic experiment pipeline for neuroscience simulations that orchestrates the full workflow:

```
Simulation → Processing → Analysis → Plotting
```

## Overview

The Pipeline handles multiple execution modes, parallel processing, batch/unified processing for consistent attractor identification, and automatic result aggregation.

### Execution Modes

| Mode | Description |
|------|-------------|
| `SINGLE` | Single simulation run |
| `REPEATED` | Multiple runs with seed management |
| `SWEEP` | Parameter sweep (one run per value) |
| `SWEEP_REPEATED` | Factorial design (sweep × repeats) |

## Quick Start

```python
from neuro_mod.pipeline import Pipeline, PipelineConfig, ExecutionMode
from neuro_mod.core.spiking_net.processing import SNNProcessor
from neuro_mod.core.spiking_net.analysis import SNNAnalyzer
from neuro_mod.execution.stagers import StageSNNSimulation

# Create component factories
def simulator_factory(seed, **kwargs):
    return StageSNNSimulation("config.yaml", random_seed=seed)

def processor_factory(raw_data, **kwargs):
    return SNNProcessor(
        spikes_path=raw_data["spikes_path"],
        clusters_path=raw_data.get("clusters_path"),
    )

# Build pipeline
pipeline = Pipeline(
    simulator_factory=simulator_factory,
    processor_factory=processor_factory,
    analyzer_factory=SNNAnalyzer,
)

# Run with configuration
result = pipeline.run(PipelineConfig(
    mode=ExecutionMode.REPEATED,
    n_repeats=10,
    parallel=True,
    save_dir=Path("experiments/run_001"),
))

# Access results
df = result.dataframes["aggregated"]
metrics = result.metrics["aggregated"]
```

## Architecture

### Protocols

The pipeline uses Python Protocols for structural typing. Any class with the required methods automatically conforms:

#### Simulator

```python
class Simulator(Protocol):
    def run(self, *args, **kwargs) -> RawOutput:
        """Execute simulation and return raw outputs."""
        ...
```

#### Processor

```python
class Processor(Protocol):
    @property
    def processed_data(self) -> ProcessedData | None: ...

    def process(self) -> ProcessedData: ...
    def save(self, path: Path) -> None: ...

    @classmethod
    def load_processed(cls, path: Path) -> ProcessedData: ...
```

#### Analyzer

```python
class Analyzer(Protocol):
    @property
    def processed_data(self) -> ProcessedData: ...

    def to_dataframe(self, *columns: str) -> pd.DataFrame: ...
    def get_summary_metrics(self) -> dict[str, Any]: ...
```

#### BatchProcessor (for unified processing)

```python
class BatchProcessor(Protocol):
    def process_batch(
        self,
        raw_outputs: list[RawOutput],
        metadata: list[dict[str, Any]],
    ) -> ProcessedData:
        """Process multiple outputs together with metadata."""
        ...
```

### Factory Pattern

Factories create component instances. The pipeline calls them with appropriate parameters:

```python
# SimulatorFactory: called with seed
simulator = simulator_factory(seed=42)

# ProcessorFactory: called with raw output
processor = processor_factory(raw_data={"spikes_path": "..."})

# AnalyzerFactory: called with processed data or path
analyzer = analyzer_factory(processed_data)

# BatchProcessorFactory: called with all outputs and metadata
batch_processor = batch_processor_factory(raw_outputs, metadata)
```

## Configuration

### PipelineConfig

```python
from neuro_mod.pipeline import PipelineConfig, ExecutionMode

config = PipelineConfig(
    # Execution mode
    mode=ExecutionMode.REPEATED,

    # Repeat settings
    n_repeats=10,
    base_seed=256,           # For deterministic seed generation
    seeds=[1, 2, 3],         # Or provide explicit seeds

    # Sweep settings (for SWEEP/SWEEP_REPEATED modes)
    sweep_param="arousal.level",
    sweep_values=[0.1, 0.5, 1.0],

    # Parallelization
    parallel=True,
    max_workers=4,
    executor="thread",       # or "process"

    # I/O settings
    save_dir=Path("experiments/run_001"),
    save_raw=False,          # Raw simulation outputs
    save_processed=True,     # Processed data
    save_analysis=True,      # DataFrames and metrics
    save_plots=True,

    # Phase toggles
    run_processing=True,
    run_analysis=True,
    run_plotting=True,

    # Logging
    log_level="INFO",
    log_to_file=True,
)
```

### Unified Processing

For REPEATED and SWEEP_REPEATED modes, all repeats are processed together to ensure consistent attractor identification across runs. This is handled automatically by the pipeline.

- **REPEATED mode**: All repeats processed together for consistent attractor IDs
- **SWEEP mode**: Each sweep value processed separately (single run per value)
- **SWEEP_REPEATED mode**: Repeats within each sweep value are unified; different sweep values are kept separate (different dynamics)

This requires a `batch_processor_factory` when using repeated modes:

```python
from neuro_mod.core.spiking_net.processing import SNNBatchProcessorFactory

pipeline = Pipeline(
    simulator_factory=...,
    processor_factory=...,
    batch_processor_factory=SNNBatchProcessorFactory(),  # Required for REPEATED/SWEEP_REPEATED
    analyzer_factory=SNNAnalyzer,
)
```

## Results

### PipelineResult

```python
result = pipeline.run(config)

# Per-run results
result.dataframes["repeat_0"]   # DataFrame for first repeat
result.metrics["repeat_0"]      # Metrics for first repeat

# Aggregated results (combined across all runs)
result.dataframes["aggregated"]
result.metrics["aggregated"]

# Time evolution data (if analyzer provides it)
result.dataframes["aggregated_time"]

# Metadata
result.seeds_used              # List of seeds used
result.duration_seconds        # Execution time
result.timestamp               # ISO timestamp
result.git_commit              # Git commit hash
```

### Loading Saved Results

```python
from neuro_mod.pipeline.io import load_result

data = load_result(Path("experiments/run_001"))
df = data["dataframes"]["aggregated"]
metrics = data["metrics"]["aggregated"]
```

## Plotting

### SeabornPlotter with PlotSpec

The standard way to add plots is with `SeabornPlotter` and `PlotSpec`:

```python
from neuro_mod.pipeline import SeabornPlotter, PlotSpec

specs = [
    PlotSpec(
        name="duration_distribution",
        plot_type="hist",
        x="duration",
        title="Attractor Duration Distribution",
        xlabel="Duration (ms)",
        ylabel="Count",
        kwargs={"bins": 40, "kde": True},
    ),
    PlotSpec(
        name="duration_over_time",
        plot_type="scatter",
        x="t_start",
        y="duration",
        hue="repeat",
        title="Duration Over Time",
        kwargs={"alpha": 0.5, "s": 10},
    ),
    PlotSpec(
        name="sweep_comparison",
        plot_type="box",
        x="sweep_value",
        y="duration",
        title="Duration by Sweep Value",
    ),
]

plotter = SeabornPlotter(specs=specs, apply_journal_style=True)
pipeline = Pipeline(..., plotter=plotter)
```

#### PlotSpec Parameters

| Parameter | Description |
|-----------|-------------|
| `name` | Unique name (used as filename) |
| `plot_type` | `"line"`, `"scatter"`, `"hist"`, `"box"`, `"violin"`, `"bar"`, `"heatmap"`, `"kde"`, `"strip"`, `"point"` |
| `x`, `y` | DataFrame column names |
| `hue` | Column for color encoding |
| `style` | Column for line style |
| `size` | Column for size encoding |
| `row`, `col` | Columns for faceting |
| `title`, `xlabel`, `ylabel` | Labels |
| `figsize` | Tuple `(width, height)` in inches |
| `kwargs` | Extra args passed to seaborn function |

#### Auto-Generated Plots

If no specs are provided, `SeabornPlotter` can auto-generate plots based on DataFrame structure:

```python
plotter = SeabornPlotter(auto_generate=True, apply_journal_style=True)
```

It detects `sweep_value`, `repeat` columns and generates appropriate visualizations.

### MatplotlibPlotter for Custom Plots

For plots requiring data manipulation or custom logic:

```python
from neuro_mod.pipeline import MatplotlibPlotter

def plot_attractor_stability(data, ax, metrics=None, **kwargs):
    """Mean duration vs occurrence count per attractor."""
    summary = data.groupby("attractor_idx").agg(
        mean_duration=("duration", "mean"),
        count=("duration", "count"),
    )
    ax.scatter(summary["count"], summary["mean_duration"], alpha=0.6)
    ax.set_xlabel("Occurrence Count")
    ax.set_ylabel("Mean Duration (ms)")
    ax.set_title("Attractor Stability")

def plot_pareto(data, ax, metrics=None, **kwargs):
    """Cumulative time coverage by attractor rank."""
    sorted_dur = data.groupby("attractor_idx")["duration"].sum()
    sorted_dur = sorted_dur.sort_values(ascending=False).cumsum()
    ax.plot(range(len(sorted_dur)), sorted_dur / sorted_dur.max())
    ax.set_xlabel("Attractor Rank")
    ax.set_ylabel("Cumulative Duration (fraction)")
    ax.set_title("Pareto Distribution")

plotter = MatplotlibPlotter(
    plot_functions=[plot_attractor_stability, plot_pareto],
    apply_journal_style=True,
)
```

The function signature is:
```python
def plot_function(data: pd.DataFrame, ax: plt.Axes, metrics: dict | None = None, **kwargs) -> None
```

### ComposablePlotter

Combine multiple plotters:

```python
from neuro_mod.pipeline import ComposablePlotter

combined = ComposablePlotter([
    SeabornPlotter(specs=[...]),
    MatplotlibPlotter(plot_functions=[...]),
])

pipeline = Pipeline(..., plotter=combined)
```

### Plotting Analyzer-Specific Data

The pipeline stores time evolution data (from `get_time_evolution_dataframe()`) in `result.dataframes["{key}_time"]`. Plot it post-pipeline:

```python
result = pipeline.run(config)

# Time evolution plotter
time_plotter = SeabornPlotter(specs=[
    PlotSpec(
        name="l2_convergence",
        plot_type="line",
        x="time_ms",
        y="transition_l2_norm",
        title="Transition Matrix Convergence",
    ),
    PlotSpec(
        name="attractor_discovery",
        plot_type="line",
        x="time_ms",
        y="unique_attractors_count",
        title="Unique Attractors Over Time",
    ),
])

# Plot time evolution data
if "aggregated_time" in result.dataframes:
    time_plotter.plot(
        result.dataframes["aggregated_time"],
        save_dir=config.save_dir / "plots",
    )
```

### Custom Plotter with Analyzer Access

For direct access to analyzer methods, create a custom plotter:

```python
from neuro_mod.pipeline.plotting import BasePlotter

class SNNAnalyzerPlotter(BasePlotter):
    def __init__(self, analyzer_factory):
        self.analyzer_factory = analyzer_factory

    def plot(self, data, metrics=None, save_dir=None, **kwargs):
        import matplotlib.pyplot as plt
        figures = []

        # Access processed_data from kwargs if pipeline provides it
        processed_data = kwargs.get("processed_data")
        if processed_data is None:
            return figures

        analyzer = self.analyzer_factory(processed_data)

        # Use analyzer methods directly
        times, norms = analyzer.get_transition_matrix_l2_norms_until_time()

        fig, ax = plt.subplots()
        ax.plot(times * 1000, norms, linewidth=2)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("L2 Norm")
        ax.set_title("Transition Matrix Evolution")
        figures.append(fig)

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / "tm_evolution.png", dpi=300)
            plt.close(fig)

        return figures
```

## Full Example: SNN Repeated Runs

```python
from pathlib import Path
from neuro_mod.pipeline import (
    Pipeline,
    PipelineConfig,
    ExecutionMode,
    SeabornPlotter,
    PlotSpec,
    ComposablePlotter,
    MatplotlibPlotter,
)
from neuro_mod.core.spiking_net.processing import SNNProcessor, SNNBatchProcessorFactory
from neuro_mod.core.spiking_net.analysis import SNNAnalyzer
from neuro_mod.execution.stagers import StageSNNSimulation


# === Component Factories ===

def create_simulator_factory(config_path: Path):
    def factory(seed: int, **kwargs):
        return StageSNNSimulation(config_path, random_seed=seed)
    return factory


def create_processor_factory():
    def factory(raw_data: dict, **kwargs):
        return SNNProcessor(
            spikes_path=raw_data["spikes_path"],
            clusters_path=raw_data.get("clusters_path"),
        )
    return factory


# === Plotters ===

def plot_stability(data, ax, metrics=None, **kwargs):
    summary = data.groupby("attractor_idx").agg(
        mean_dur=("duration", "mean"),
        count=("duration", "count"),
    )
    ax.scatter(summary["count"], summary["mean_dur"], alpha=0.6, s=20)
    ax.set_xlabel("Occurrences")
    ax.set_ylabel("Mean Duration (ms)")
    ax.set_title("Attractor Stability")


main_plotter = SeabornPlotter(
    specs=[
        PlotSpec(
            name="duration_hist",
            plot_type="hist",
            x="duration",
            title="Duration Distribution",
            kwargs={"bins": 40, "kde": True},
        ),
        PlotSpec(
            name="duration_scatter",
            plot_type="scatter",
            x="t_start",
            y="duration",
            hue="repeat",
            title="Duration Over Time",
            kwargs={"alpha": 0.4, "s": 8},
        ),
    ],
    apply_journal_style=True,
)

custom_plotter = MatplotlibPlotter(
    plot_functions=[plot_stability],
    apply_journal_style=True,
)

combined_plotter = ComposablePlotter([main_plotter, custom_plotter])


# === Pipeline ===

pipeline = Pipeline(
    simulator_factory=create_simulator_factory(Path("configs/snn.yaml")),
    processor_factory=create_processor_factory(),
    batch_processor_factory=SNNBatchProcessorFactory(),
    analyzer_factory=SNNAnalyzer,
    plotter=combined_plotter,
)

config = PipelineConfig(
    mode=ExecutionMode.REPEATED,
    n_repeats=10,
    base_seed=42,
    parallel=True,
    max_workers=4,
    save_dir=Path("experiments/snn_repeated"),
    log_level="INFO",
)

result = pipeline.run(config)


# === Post-Processing ===

print(f"Completed in {result.duration_seconds:.1f}s")
print(f"Seeds: {result.seeds_used}")

df = result.dataframes["aggregated"]
print(f"Total occurrences: {len(df)}")
print(f"Unique attractors: {df['attractor_idx'].nunique()}")

# Plot time evolution
if "aggregated_time" in result.dataframes:
    time_plotter = SeabornPlotter(specs=[
        PlotSpec(name="l2_norm", plot_type="line", x="time_ms", y="transition_l2_norm"),
    ])
    time_plotter.plot(
        result.dataframes["aggregated_time"],
        save_dir=config.save_dir / "plots",
    )
```

## Directory Structure

After running with `save_dir`:

```
experiments/run_001/
├── metadata/
│   ├── pipeline_config.json
│   ├── pipeline_metadata.json
│   └── seeds.npy
├── dataframes/
│   ├── repeat_0.parquet
│   ├── repeat_1.parquet
│   ├── aggregated.parquet
│   └── aggregated_time.parquet
├── metrics/
│   ├── repeat_0.json
│   ├── repeat_1.json
│   └── aggregated.json
├── processed/
│   └── unified/
│       ├── attractors.npy
│       └── batch_config.json
├── plots/
│   ├── duration_hist.png
│   ├── duration_scatter.png
│   └── ...
└── pipeline.log
```

`batch_config.json` includes timing metadata for unified runs:

```json
{
  "dt": 0.0005,
  "total_duration_ms": 123456.0,
  "n_runs": 2,
  "repeats": [
    {"repeat_idx": 0, "duration_ms": 12345.0, "seed": 101},
    {"repeat_idx": 1, "duration_ms": 12345.0, "seed": 102}
  ]
}
```

## API Reference

### Pipeline

```python
Pipeline(
    simulator_factory: SimulatorFactory,
    processor_factory: ProcessorFactory | None = None,
    batch_processor_factory: BatchProcessorFactory | None = None,
    analyzer_factory: AnalyzerFactory | None = None,
    plotter: Plotter | None = None,
    logger: logging.Logger | None = None,
)

Pipeline.run(config: PipelineConfig) -> PipelineResult
```

### I/O Utilities

```python
from neuro_mod.pipeline.io import (
    save_result,
    load_result,
    save_dataframe,
    load_dataframe,
    save_metrics,
    load_metrics,
    get_git_commit,
    get_timestamp,
)
```
