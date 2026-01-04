#!/usr/bin/env python
"""Run repeated SNN simulations using the Pipeline architecture.

This script runs SNN simulations with full reproducibility, processing,
and optional unified attractor identification across repeats.

Usage:
    python scripts/run_snn.py --config configs/snn_long_run.yaml --n-repeats 10

    # With unified processing (consistent attractor IDs across repeats)
    python scripts/run_snn.py --config configs/snn_long_run.yaml --n-repeats 10 --unified

    # Parallel execution
    python scripts/run_snn.py --config configs/snn_long_run.yaml --n-repeats 10 --parallel
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from neuro_mod.pipeline import (
    ExecutionMode,
    Pipeline,
    PipelineConfig,
    SeabornPlotter,
    PlotSpec,
)
from neuro_mod.core.spiking_net.analysis import SNNAnalyzer
from neuro_mod.core.spiking_net.processing import (
    SNNProcessor,
    SNNBatchProcessorFactory,
)
from neuro_mod.execution.stagers import StageSNNSimulation
from neuro_mod.execution.helpers.cli import resolve_path, save_cmd
from neuro_mod.visualization import folder_plots_to_pdf


def _build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run repeated SNN simulations with the Pipeline architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default=str(root / "configs/snn_long_run.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(root / "simulations/snn_long_run"),
        help="Output directory for simulation artifacts.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=2,
        help="Number of repeats to run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=256,
        help="Base seed for reproducible seed generation.",
    )
    parser.add_argument(
        "--seeds-file",
        default=None,
        help="Optional path to a seeds.txt file to load explicit seeds.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run repeats in parallel.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of workers for parallel execution.",
    )
    parser.add_argument(
        "--executor",
        choices=("thread", "process"),
        default="thread",
        help="Parallel executor backend.",
    )
    parser.add_argument(
        "--unified",
        action="store_true",
        default=True,
        help="Use unified processing for consistent attractor IDs across repeats.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--raster-plots",
        action="store_true",
        help="Save per-run raster plots to save_dir/plots/rasters.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw spike data after processing (default: delete to save space).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level.",
    )
    return parser


class _RasterPlotRunner:
    """Wrap a stager to optionally save per-run raster plots."""

    def __init__(self, stager: StageSNNSimulation, seed: int, save_dir: Path) -> None:
        self._stager = stager
        self._seed = seed
        self._save_dir = save_dir

    def run(self) -> dict[str, Any]:
        outputs = self._stager.run()
        spikes = outputs.get("spikes")
        if spikes is not None and hasattr(self._stager, "_plot"):
            plot_dir = self._save_dir / "plots" / "rasters"
            plot_dir.mkdir(parents=True, exist_ok=True)
            self._stager._plot(spikes, plt_path=plot_dir / f"spikes_seed_{self._seed}.png")
        return outputs


def create_simulator_factory(
    config_path: Path,
    *,
    raster_plots: bool,
    save_dir: Path,
):
    """Create a factory that produces SNN simulators with different seeds."""

    def factory(seed: int, **kwargs) -> StageSNNSimulation | _RasterPlotRunner:
        stager = StageSNNSimulation(config_path, random_seed=seed)
        if raster_plots:
            return _RasterPlotRunner(stager, seed, save_dir)
        return stager

    return factory


def create_processor_factory(dt: float = 0.5e-3):
    """Create a factory that produces SNNProcessor instances."""

    def factory(raw_data: dict[str, Any], **kwargs) -> SNNProcessor:
        spikes_path = raw_data.get("spikes_path")
        clusters_path = raw_data.get("clusters_path")

        if spikes_path is None:
            raise ValueError(
                "Pipeline requires simulation outputs to be saved. "
                "Ensure config has settings.save: true"
            )

        return SNNProcessor(
            spikes_path=spikes_path,
            clusters_path=clusters_path,
            dt=raw_data.get("dt", dt),
        )

    return factory


def create_plotter() -> SeabornPlotter:
    """Create a SeabornPlotter with SNN-specific plot specifications."""
    specs = [
        PlotSpec(
            name="duration_distribution",
            plot_type="hist",
            x="duration_ms",
            title="Attractor Duration Distribution",
            xlabel="Duration (ms)",
            ylabel="Count",
            kwargs={"bins": 40, "kde": True, "color": "#2a9d8f"},
        ),
        PlotSpec(
            name="duration_over_time",
            plot_type="scatter",
            x="start",
            y="duration_ms",
            title="Attractor Duration Over Time",
            xlabel="Start Time (s)",
            ylabel="Duration (ms)",
            kwargs={"alpha": 0.3, "s": 10},
        ),
        PlotSpec(
            name="attractor_counts",
            plot_type="hist",
            x="attractor_id",
            title="Attractor Occurrence Counts",
            xlabel="Attractor ID",
            ylabel="Count",
            kwargs={"bins": 30, "color": "#e76f51"},
        ),
    ]
    return SeabornPlotter(specs=specs, apply_journal_style=True)


def create_time_plotter() -> SeabornPlotter:
    """Create a SeabornPlotter for time-evolution SNN metrics."""
    specs = [
        PlotSpec(
            name="transition_l2_norm_over_time",
            plot_type="line",
            x="time_ms",
            y="transition_l2_norm",
            title="Transition Matrix L2 Norm Over Time",
            xlabel="Time (ms)",
            ylabel="L2 Norm",
            kwargs={"linewidth": 2},
        ),
        PlotSpec(
            name="unique_attractors_over_time",
            plot_type="line",
            x="time_ms",
            y="unique_attractors_count",
            title="Unique Attractors Over Time",
            xlabel="Time (ms)",
            ylabel="Unique Attractors",
            kwargs={"linewidth": 2},
        ),
    ]
    return SeabornPlotter(specs=specs, apply_journal_style=True)


def load_seeds_from_file(seeds_file: Path) -> list[int]:
    """Load seeds from a text file (one seed per line)."""
    with open(seeds_file) as f:
        return [int(line.strip()) for line in f if line.strip()]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    args = _build_parser(root).parse_args()

    config_path = resolve_path(root, args.config)
    save_dir = resolve_path(root, args.save_dir)

    # Save command for reproducibility
    save_cmd(save_dir / "metadata")

    # Load explicit seeds if provided
    seeds = None
    if args.seeds_file:
        seeds_file = resolve_path(root, args.seeds_file)
        if seeds_file.exists():
            seeds = load_seeds_from_file(seeds_file)
            print(f"Loaded {len(seeds)} seeds from {seeds_file}")

    # Create pipeline components
    simulator_factory = create_simulator_factory(
        config_path,
        raster_plots=args.raster_plots,
        save_dir=save_dir,
    )
    processor_factory = create_processor_factory()
    batch_processor_factory = SNNBatchProcessorFactory() if args.unified else None
    plotter = None if args.no_plots else create_plotter()
    time_plotter = None if args.no_plots else create_time_plotter()

    # Create pipeline
    pipeline = Pipeline(
        simulator_factory=simulator_factory,
        processor_factory=processor_factory,
        batch_processor_factory=batch_processor_factory,
        analyzer_factory=SNNAnalyzer,
        plotter=plotter,
    )

    # Build pipeline config
    config = PipelineConfig(
        mode=ExecutionMode.REPEATED,
        n_repeats=args.n_repeats,
        base_seed=args.seed,
        seeds=seeds,
        parallel=args.parallel,
        max_workers=args.max_workers,
        executor=args.executor,
        save_dir=save_dir,
        save_raw=True,
        save_processed=True,
        save_analysis=True,
        save_plots=not args.no_plots,
        unified_processing=args.unified,
        log_level=args.log_level,
    )

    # Run pipeline
    result = pipeline.run(config)

    # Merge raster plots into a single PDF
    if args.raster_plots:
        raster_dir = save_dir / "plots" / "rasters"
        if raster_dir.is_dir():
            try:
                folder_plots_to_pdf(
                    raster_dir,
                    output_path=save_dir / "plots" / "rasters.pdf",
                )
            except ValueError as exc:
                print(f"Skipping raster PDF export: {exc}")

    # Merge pipeline plots into a single PDF
    if not args.no_plots:
        plots_dir = save_dir / "plots"
        if plots_dir.is_dir():
            try:
                folder_plots_to_pdf(
                    plots_dir,
                    output_path=save_dir / "plots" / "plots.pdf",
                )
            except ValueError as exc:
                print(f"Skipping plots PDF export: {exc}")

    # Plot time-evolution metrics if available
    if time_plotter is not None:
        time_key = "aggregated_time" if "aggregated_time" in result.dataframes else None
        if time_key is not None:
            time_plotter.plot(
                data=result.dataframes[time_key],
                metrics=result.metrics.get("aggregated"),
                save_dir=save_dir / "plots",
            )

    # Print summary
    print("\n" + "=" * 60)
    print("SNN SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Seeds used: {len(result.seeds_used)} seeds")
    print(f"Results saved to: {save_dir}")

    if "aggregated" in result.dataframes:
        df = result.dataframes["aggregated"]
        print(f"Aggregated DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
        if "repeat" in df.columns:
            print(f"Unique repeats: {df['repeat'].nunique()}")

    if "aggregated" in result.metrics:
        metrics = result.metrics["aggregated"]
        print("\nAggregated Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # Clean up raw data if not keeping it
    if not args.keep_raw:
        raw_data_dir = save_dir / "data"
        if raw_data_dir.exists():
            print(f"\nRemoving raw data directory: {raw_data_dir}")
            shutil.rmtree(raw_data_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
