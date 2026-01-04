#!/usr/bin/env python
"""Self-contained demo of the Pipeline architecture.

This script demonstrates the Pipeline workflow using simple mock components,
showing how the pieces fit together without running actual simulations.

Run with:
    python scripts/pipeline_demo.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from neuro_mod.pipeline import (
    ExecutionMode,
    Pipeline,
    PipelineConfig,
    PlotSpec,
    SeabornPlotter,
)


# =============================================================================
# Mock Components (replace with real ones in production)
# =============================================================================


@dataclass
class MockSimulationOutput:
    """Mock output from a simulation."""
    spikes: np.ndarray
    n_attractors: int
    seed: int


class MockSimulator:
    """Mock simulator that generates random attractor data."""

    def __init__(self, seed: int, n_attractors: int = 10, n_occurrences: int = 100):
        self.seed = seed
        self.n_attractors = n_attractors
        self.n_occurrences = n_occurrences
        self.rng = np.random.default_rng(seed)

    def run(self) -> MockSimulationOutput:
        """Run mock simulation."""
        # Generate random "spike" data
        spikes = self.rng.random((self.n_occurrences, 10))
        return MockSimulationOutput(
            spikes=spikes,
            n_attractors=self.n_attractors,
            seed=self.seed,
        )


class MockProcessor:
    """Mock processor that extracts attractor data."""

    def __init__(self, raw_data: MockSimulationOutput):
        self.raw_data = raw_data
        self._processed_data: dict | None = None
        self.rng = np.random.default_rng(raw_data.seed)

    @property
    def processed_data(self) -> dict | None:
        return self._processed_data

    def process(self) -> dict:
        """Process raw data into attractors."""
        n_occurrences = len(self.raw_data.spikes)

        # Generate mock attractor occurrences
        attractor_ids = self.rng.integers(0, self.raw_data.n_attractors, n_occurrences)
        starts = np.sort(self.rng.uniform(0, 100, n_occurrences))
        durations = self.rng.exponential(5, n_occurrences)  # Exponential durations

        self._processed_data = {
            "attractor_ids": attractor_ids,
            "starts": starts,
            "durations": durations,
            "n_attractors": self.raw_data.n_attractors,
            "seed": self.raw_data.seed,
        }
        return self._processed_data

    def save(self, path: Path) -> None:
        """Save processed data."""
        path.mkdir(parents=True, exist_ok=True)
        if self._processed_data:
            np.savez(
                path / "processed.npz",
                **{k: v for k, v in self._processed_data.items() if isinstance(v, np.ndarray)},
            )

    @classmethod
    def load_processed(cls, path: Path) -> dict:
        """Load processed data."""
        data = np.load(path / "processed.npz")
        return dict(data)


class MockAnalyzer:
    """Mock analyzer that computes statistics."""

    def __init__(self, processed_data: dict | Path):
        if isinstance(processed_data, Path):
            processed_data = MockProcessor.load_processed(processed_data)
        self._processed_data = processed_data

    @property
    def processed_data(self) -> dict:
        return self._processed_data

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        df = pd.DataFrame({
            "attractor_id": self._processed_data["attractor_ids"],
            "start": self._processed_data["starts"],
            "duration_ms": self._processed_data["durations"],
        })
        # Add metadata columns if present (from batch processing)
        if "repeat_indices" in self._processed_data:
            df["repeat"] = self._processed_data["repeat_indices"]
        if "seeds" in self._processed_data:
            df["seed"] = self._processed_data["seeds"]
        if "sweep_values" in self._processed_data:
            df["sweep_value"] = self._processed_data["sweep_values"]
        return df

    def get_summary_metrics(self) -> dict:
        """Compute summary metrics."""
        durations = self._processed_data["durations"]
        return {
            "n_occurrences": len(durations),
            "n_attractors": self._processed_data["n_attractors"],
            "mean_duration_ms": float(np.mean(durations)),
            "std_duration_ms": float(np.std(durations)),
            "total_duration_ms": float(np.sum(durations)),
        }


class MockBatchProcessor:
    """Mock batch processor for unified processing of multiple runs."""

    def __init__(
        self,
        raw_outputs: list[MockSimulationOutput],
        metadata: list[dict],
    ):
        self.raw_outputs = raw_outputs
        self.metadata = metadata
        self._processed_data: dict | None = None

    @property
    def processed_data(self) -> dict | None:
        return self._processed_data

    def process_batch(
        self,
        raw_outputs: list[MockSimulationOutput],
        metadata: list[dict],
    ) -> dict:
        """Process all outputs together for consistent attractor IDs."""
        # Combine all data from all runs
        all_attractor_ids = []
        all_starts = []
        all_durations = []
        all_repeat_indices = []
        all_seeds = []
        all_sweep_values = []

        cumulative_time = 0.0
        total_n_attractors = 0

        for raw, meta in zip(raw_outputs, metadata):
            # Process each run
            rng = np.random.default_rng(raw.seed)
            n_occurrences = len(raw.spikes)

            # Generate mock attractor data
            attractor_ids = rng.integers(0, raw.n_attractors, n_occurrences)
            starts = np.sort(rng.uniform(0, 100, n_occurrences)) + cumulative_time
            durations = rng.exponential(5, n_occurrences)

            all_attractor_ids.extend(attractor_ids.tolist())
            all_starts.extend(starts.tolist())
            all_durations.extend(durations.tolist())

            # Embed metadata for each occurrence
            all_repeat_indices.extend([meta.get("repeat_idx")] * n_occurrences)
            all_seeds.extend([meta.get("seed")] * n_occurrences)
            all_sweep_values.extend([meta.get("sweep_value")] * n_occurrences)

            cumulative_time = starts[-1] + 10  # Gap between runs
            total_n_attractors = max(total_n_attractors, raw.n_attractors)

        self._processed_data = {
            "attractor_ids": np.array(all_attractor_ids),
            "starts": np.array(all_starts),
            "durations": np.array(all_durations),
            "repeat_indices": all_repeat_indices,
            "seeds": all_seeds,
            "sweep_values": all_sweep_values,
            "n_attractors": total_n_attractors,
        }
        return self._processed_data

    def save(self, path: Path) -> None:
        """Save processed data."""
        path.mkdir(parents=True, exist_ok=True)
        if self._processed_data:
            np.savez(path / "batch_processed.npz", **self._processed_data)

    @classmethod
    def load_processed(cls, path: Path) -> dict:
        """Load processed data."""
        return dict(np.load(path / "batch_processed.npz", allow_pickle=True))


class MockBatchProcessorFactory:
    """Factory for creating MockBatchProcessor instances."""

    def __call__(
        self,
        raw_outputs: list[MockSimulationOutput],
        metadata: list[dict],
        **kwargs,
    ) -> MockBatchProcessor:
        return MockBatchProcessor(raw_outputs, metadata)

    @property
    def supports_batch(self) -> bool:
        return True


# =============================================================================
# Factory Functions
# =============================================================================


def mock_simulator_factory(seed: int, **kwargs) -> MockSimulator:
    """Factory to create mock simulators."""
    return MockSimulator(seed=seed, n_attractors=15, n_occurrences=200)


def mock_processor_factory(raw_data: MockSimulationOutput, **kwargs) -> MockProcessor:
    """Factory to create mock processors."""
    return MockProcessor(raw_data)


# =============================================================================
# Custom Plotter
# =============================================================================


def create_demo_plotter() -> SeabornPlotter:
    """Create a plotter with demo-appropriate plot specs."""
    specs = [
        PlotSpec(
            name="duration_histogram",
            plot_type="hist",
            x="duration_ms",
            title="Attractor Duration Distribution",
            xlabel="Duration (ms)",
            ylabel="Count",
            kwargs={"bins": 25, "kde": True, "color": "#2a9d8f"},
        ),
        PlotSpec(
            name="duration_over_time",
            plot_type="scatter",
            x="start",
            y="duration_ms",
            title="Duration vs Start Time",
            xlabel="Start Time (s)",
            ylabel="Duration (ms)",
            kwargs={"alpha": 0.5, "s": 30},
        ),
        PlotSpec(
            name="attractor_counts",
            plot_type="hist",
            x="attractor_id",
            title="Attractor Occurrence Counts",
            xlabel="Attractor ID",
            ylabel="Count",
            kwargs={"bins": 15, "color": "#e76f51"},
        ),
    ]
    return SeabornPlotter(specs=specs, apply_journal_style=True)


# =============================================================================
# Demo Functions
# =============================================================================


def demo_single_run():
    """Demonstrate single run mode."""
    print("\n" + "=" * 60)
    print("DEMO: Single Run")
    print("=" * 60)

    pipeline = Pipeline(
        simulator_factory=mock_simulator_factory,
        processor_factory=mock_processor_factory,
        analyzer_factory=MockAnalyzer,
        plotter=create_demo_plotter(),
    )

    result = pipeline.run(PipelineConfig(
        mode=ExecutionMode.SINGLE,
        base_seed=42,
        save_dir=REPO_ROOT / "experiments" / "demo_single",
        log_level="INFO",
    ))

    print(f"\nResults:")
    print(f"  Duration: {result.duration_seconds:.2f}s")
    print(f"  DataFrame shape: {result.dataframes['single'].shape}")
    print(f"  Metrics: {result.metrics['single']}")
    print(f"  Figures: {len(result.figures)}")


def demo_repeated_runs():
    """Demonstrate repeated runs mode."""
    print("\n" + "=" * 60)
    print("DEMO: Repeated Runs (5 repeats)")
    print("=" * 60)

    pipeline = Pipeline(
        simulator_factory=mock_simulator_factory,
        processor_factory=mock_processor_factory,
        analyzer_factory=MockAnalyzer,
        plotter=SeabornPlotter(auto_generate=True),  # Auto-generate plots
    )

    result = pipeline.run(PipelineConfig(
        mode=ExecutionMode.REPEATED,
        n_repeats=5,
        base_seed=123,
        save_dir=REPO_ROOT / "experiments" / "demo_repeated",
        log_level="INFO",
    ))

    print(f"\nResults:")
    print(f"  Duration: {result.duration_seconds:.2f}s")
    print(f"  Seeds used: {result.seeds_used}")
    print(f"  Aggregated DataFrame shape: {result.dataframes['aggregated'].shape}")
    print(f"  Aggregated metrics: {result.metrics['aggregated']}")


def demo_sweep():
    """Demonstrate parameter sweep mode."""
    print("\n" + "=" * 60)
    print("DEMO: Parameter Sweep")
    print("=" * 60)

    # For sweep, we need a factory that accepts sweep parameters
    def sweep_simulator_factory(seed: int, sweep_param=None, sweep_value=None, **kwargs):
        # In a real scenario, sweep_value would modify simulation parameters
        n_attractors = int(sweep_value) if sweep_value else 10
        return MockSimulator(seed=seed, n_attractors=n_attractors, n_occurrences=150)

    pipeline = Pipeline(
        simulator_factory=sweep_simulator_factory,
        processor_factory=mock_processor_factory,
        analyzer_factory=MockAnalyzer,
        plotter=SeabornPlotter(auto_generate=True),
    )

    result = pipeline.run(PipelineConfig(
        mode=ExecutionMode.SWEEP,
        sweep_param="n_attractors",
        sweep_values=[5, 10, 15, 20],
        base_seed=456,
        save_dir=REPO_ROOT / "experiments" / "demo_sweep",
        log_level="INFO",
    ))

    print(f"\nResults:")
    print(f"  Duration: {result.duration_seconds:.2f}s")
    print(f"  Sweep values: {result.sweep_metadata['values']}")
    print(f"  Aggregated DataFrame shape: {result.dataframes['aggregated'].shape}")
    print(f"  Columns: {list(result.dataframes['aggregated'].columns)}")


def demo_sweep_repeated():
    """Demonstrate sweep with repeats mode (per-run processing)."""
    print("\n" + "=" * 60)
    print("DEMO: Sweep + Repeated (4 values x 3 repeats) - Per-Run Processing")
    print("=" * 60)

    def sweep_simulator_factory(seed: int, sweep_param=None, sweep_value=None, **kwargs):
        n_attractors = int(sweep_value) if sweep_value else 10
        return MockSimulator(seed=seed, n_attractors=n_attractors, n_occurrences=100)

    pipeline = Pipeline(
        simulator_factory=sweep_simulator_factory,
        processor_factory=mock_processor_factory,
        analyzer_factory=MockAnalyzer,
        plotter=SeabornPlotter(auto_generate=True),
    )

    result = pipeline.run(PipelineConfig(
        mode=ExecutionMode.SWEEP_REPEATED,
        n_repeats=3,
        sweep_param="n_attractors",
        sweep_values=[5, 10, 15, 20],
        base_seed=789,
        unified_processing=False,  # Per-run processing (old behavior)
        parallel=False,
        save_dir=REPO_ROOT / "experiments" / "demo_sweep_repeated",
        log_level="INFO",
    ))

    print(f"\nResults:")
    print(f"  Duration: {result.duration_seconds:.2f}s")
    print(f"  Total runs: {len(result.seeds_used)} seeds x {len(result.sweep_metadata['values'])} sweep values")
    print(f"  Aggregated DataFrame shape: {result.dataframes['aggregated'].shape}")
    print(f"  Sample of aggregated metrics:")
    for k, v in list(result.metrics['aggregated'].items())[:5]:
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")


def demo_unified_repeated():
    """Demonstrate unified processing for repeated runs."""
    print("\n" + "=" * 60)
    print("DEMO: Repeated Runs with UNIFIED Processing (5 repeats)")
    print("=" * 60)
    print("All repeats processed together for consistent attractor IDs.")

    pipeline = Pipeline(
        simulator_factory=mock_simulator_factory,
        processor_factory=mock_processor_factory,
        batch_processor_factory=MockBatchProcessorFactory(),  # Enable batch processing
        analyzer_factory=MockAnalyzer,
        plotter=SeabornPlotter(auto_generate=True),
    )

    result = pipeline.run(PipelineConfig(
        mode=ExecutionMode.REPEATED,
        n_repeats=5,
        base_seed=111,
        unified_processing=True,  # Process all repeats together
        save_dir=REPO_ROOT / "experiments" / "demo_unified_repeated",
        log_level="INFO",
    ))

    print(f"\nResults:")
    print(f"  Duration: {result.duration_seconds:.2f}s")
    print(f"  Seeds used: {result.seeds_used}")
    agg_df = result.dataframes.get("aggregated")
    if agg_df is not None:
        print(f"  Aggregated DataFrame shape: {agg_df.shape}")
        print(f"  Columns: {list(agg_df.columns)}")
        if "repeat" in agg_df.columns:
            print(f"  Unique repeats in data: {sorted(agg_df['repeat'].dropna().unique())}")


def demo_unified_sweep_repeated():
    """Demonstrate unified processing for sweep with repeats.

    Hierarchy: sweep_value -> attractors
    - Different sweep values have different dynamics (separate attractor sets)
    - Repeats within each sweep value are unified (consistent IDs)
    """
    print("\n" + "=" * 60)
    print("DEMO: Sweep + Repeated with UNIFIED Processing")
    print("=" * 60)
    print("Hierarchy: sweep_value -> attractors")
    print("  - Each sweep value processed separately (different dynamics)")
    print("  - Repeats within each sweep value unified (consistent IDs)")

    def sweep_simulator_factory(seed: int, sweep_param=None, sweep_value=None, **kwargs):
        n_attractors = int(sweep_value) if sweep_value else 10
        return MockSimulator(seed=seed, n_attractors=n_attractors, n_occurrences=100)

    pipeline = Pipeline(
        simulator_factory=sweep_simulator_factory,
        processor_factory=mock_processor_factory,
        batch_processor_factory=MockBatchProcessorFactory(),  # Enable batch processing
        analyzer_factory=MockAnalyzer,
        plotter=SeabornPlotter(auto_generate=True),
    )

    result = pipeline.run(PipelineConfig(
        mode=ExecutionMode.SWEEP_REPEATED,
        n_repeats=3,
        sweep_param="n_attractors",
        sweep_values=[5, 10, 15],
        base_seed=222,
        unified_processing=True,  # Unified per sweep value
        save_dir=REPO_ROOT / "experiments" / "demo_unified_sweep_repeated",
        log_level="INFO",
    ))

    print(f"\nResults:")
    print(f"  Duration: {result.duration_seconds:.2f}s")
    print(f"  Sweep values: {result.sweep_metadata['values']}")
    agg_df = result.dataframes.get("aggregated")
    if agg_df is not None:
        print(f"  Aggregated DataFrame shape: {agg_df.shape}")
        print(f"  Columns: {list(agg_df.columns)}")
        if "sweep_value" in agg_df.columns:
            print(f"  Unique sweep values: {sorted(agg_df['sweep_value'].dropna().unique())}")
        if "repeat" in agg_df.columns:
            print(f"  Unique repeats: {sorted(agg_df['repeat'].dropna().unique())}")


def main():
    """Run all demos."""
    print("=" * 60)
    print("PIPELINE ARCHITECTURE DEMO")
    print("=" * 60)
    print("\nThis demo shows the Pipeline workflow using mock components.")
    print("Replace MockSimulator/Processor/Analyzer with real ones for production.")

    # Run demos - basic modes
    demo_single_run()
    demo_repeated_runs()
    demo_sweep()
    demo_sweep_repeated()

    # Unified processing demos
    demo_unified_repeated()
    demo_unified_sweep_repeated()

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {REPO_ROOT / 'experiments'}")
    print("Check the 'plots' subdirectory in each experiment folder for figures.")
    print("\nUnified processing ensures consistent attractor IDs across repeats.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
