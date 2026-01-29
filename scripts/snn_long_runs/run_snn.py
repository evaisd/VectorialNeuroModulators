#!/usr/bin/env python
"""Run repeated SNN simulations using the Pipeline architecture.

This script runs SNN simulations with full reproducibility, processing,
and unified attractor identification across repeats.

Usage:
    python scripts/run_snn.py --config configs/snn_long_run.yaml --n-repeats 10

    # Parallel execution
    python scripts/run_snn.py --config configs/snn_long_run.yaml --n-repeats 10 --parallel
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from neuro_mod.pipeline import (
    ComposablePlotter,
    ExecutionMode,
    Pipeline,
    PipelineConfig,
    PlotSpec,
    SeabornPlotter,
    SpecPlotter,
)
from neuro_mod.core.spiking_net.analysis import SNNAnalyzer
from neuro_mod.core.spiking_net.processing import (
    SNNProcessor,
    SNNBatchProcessorFactory,
)
import yaml

from neuro_mod.execution.stagers import StageSNNSimulation
from neuro_mod.execution.helpers.cli import resolve_path, save_cmd
from neuro_mod.visualization import folder_plots_to_pdf
from neuro_mod.analysis import MetricResult, metric, manipulation
import pandas as pd

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
        "--style",
        default=str(root / "style/neuroips.mplstyle"),
        help="Matplotlib style name or path to a .mplstyle file.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=125,
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
    parser.add_argument(
        "--verbose-memory",
        action="store_true",
        help="Log memory usage after each repeat (requires psutil for detailed info).",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Skip simulation, process existing data files from save_dir/data.",
    )
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument(
        "--time-dt-ms",
        type=float,
        default=None,
        help="Time step in ms for time evolution dataframe density.",
    )
    time_group.add_argument(
        "--time-steps",
        type=int,
        default=200,
        help="Number of time steps for time evolution dataframe density.",
    )
    return parser


def _apply_style(style: str, root: Path) -> None:
    style_path = Path(style)
    if not style_path.is_absolute():
        style_path = root / style
    try:
        if style_path.exists():
            plt.style.use(str(style_path))
        else:
            plt.style.use(style)
    except OSError as exc:
        print(f"Warning: failed to apply style '{style}': {exc}")


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


# =============================================================================
# Color Palette
# =============================================================================

COLORS = {
    "primary": "#0072b2",
    "secondary": "#e69f00",
    "tertiary": "#009e73",
    "accent": "#d55e00",
    "neutral": "#f0e442",
    "bar": "#56b4e9",
    "scatter": "#cc79a7",
}


# =============================================================================
# Spec-driven Plotter Factory
# =============================================================================


class _ExpSNNAnalyzer(SNNAnalyzer):

    @manipulation("num_clusters")
    def agg_by_num_clusters(self) -> pd.DataFrame:
        df = self.per_attractor()
        agg = df.groupby('num_clusters').agg(
            total_occurrences=("occurrences", "sum"),
            mean_duration=("mean_duration", "mean"),
            std_duration=("std_duration", "mean"),
            first_start=("first_start", "count"),
        ).loc[1:]
        return agg

    @metric("01_occurrences_vs_lifespan_scatter", expects="per_attractor")
    def occ_vs_life(self, df: pd.DataFrame) -> MetricResult:
        return MetricResult(
            x=df["occurrences"].to_numpy(),
            y=df["mean_duration"].to_numpy(),
            labels=df["num_clusters"].to_numpy(),
            metadata={"std_duration": df["std_duration"].to_numpy()},
        )

    @metric("02_occurrence_duration_hist")
    def occ_duration(self, df: pd.DataFrame) -> MetricResult:
        return MetricResult(
            x=df["duration"].to_numpy(),
            y=None,
        )

    @metric("03_size_vs_occurrence_violin", expects="per_attractor")
    def size_vs_occurrence(self, df: pd.DataFrame) -> MetricResult:
        return MetricResult(
            x=df["num_clusters"].to_numpy(),
            y=df["occurrences"].to_numpy(),
        )

    @metric("04_size_vs_lifespan_violin", expects="per_attractor")
    def size_vs_lifespan(self, df: pd.DataFrame) -> MetricResult:
        return MetricResult(
            x=df["num_clusters"].to_numpy(),
            y=df["mean_duration"].to_numpy(),
        )

    @metric("05_lifespan_distribution_hist", expects="per_attractor")
    def lifespan_distribution(self, df: pd.DataFrame) -> MetricResult:
        return MetricResult(
            x=df["mean_duration"],
            y=None
        )

    @metric("06_mean_lifespan_by_size_bar", expects="num_clusters")
    def mean_lifespan_by_size(self, df: pd.DataFrame) -> MetricResult:
        return MetricResult(
            x=df.reset_index()['num_clusters'].to_numpy(),
            y=df["mean_duration"].to_numpy(),
        )

    @metric("07_discovery_by_num_clusters_line", expects="per_attractor")
    def discovery_by_num_clusters(
            self,
            df: pd.DataFrame,
            *,
            total_clusters: int = 18,
            skip_zero: bool = True,
    ) -> MetricResult:
        from math import comb
        import numpy as np
        total_duration_s = self.total_duration_ms / 1e3
        if total_duration_s == 0 and not df.empty:
            total_duration_s = float(df["last_end"].max())

        xs, ys, labels = [], [], []
        for k, g in sorted(df.groupby("num_clusters")):
            if skip_zero and k == 0:
                continue
            g = g.sort_values("first_start")
            x = g["first_start"].to_numpy() / total_duration_s
            y = (np.arange(len(g)) + 1) / comb(total_clusters, k)
            xs.append(x)
            ys.append(y)
            labels.append(np.full_like(x, k, dtype=int))

        if not xs:
            return MetricResult(x=np.array([]), y=np.array([]))

        return MetricResult(
            x=np.concatenate(xs),
            y=np.concatenate(ys),
            labels=np.concatenate(labels),
            metadata={"legend_title": "# clusters", "reference_line": ((0, 1), (1, 1))},
        )

    @metric("08_discovery_rate_line", expects="time_evolution")
    def discovery_rate(
        self,
        df: pd.DataFrame,
        *,
        time_unit: str = "ms",
    ) -> MetricResult:
        times = df["time_ms"].to_numpy()
        if time_unit == "s":
            times = times / 1e3
        elif time_unit == "min":
            times = times / 6e4
        return MetricResult(
            x=times,
            y=df["discovery_rate_per_s"].to_numpy(),
            metadata={"time_unit": time_unit},
        )

    @metric("09_cumulative_attractors_line", expects="time_evolution")
    def cumulative_attractors(
        self,
        df: pd.DataFrame,
        *,
        time_unit: str = "ms",
    ) -> MetricResult:
        times = df["time_ms"].to_numpy()
        if time_unit == "s":
            times = times / 1e3
        elif time_unit == "min":
            times = times / 6e4
        return MetricResult(
            x=times,
            y=df["unique_attractors_count"].to_numpy(),
            metadata={"time_unit": time_unit},
        )

    @metric("10_tpm_L2_norm_vs_time_line", expects="time_evolution")
    def tpm_l2_norm_vs_time(
            self,
            df: pd.DataFrame,
            *,
            time_unit: str = "ms"
    ) -> MetricResult:
        times = df["time_ms"].to_numpy()
        if time_unit == "s":
            times = times / 1e3
        elif time_unit == "min":
            times = times / 6e4
        return MetricResult(
            x=times,
            y=df["transition_l2_norm"].to_numpy(),
        )

def create_plotter(
    *,
    time_dt_ms: float | None,
    time_steps: int | None,
) -> ComposablePlotter:
    """Create plotters for analyzer-driven plots."""
    time_kwargs: dict[str, Any] = {}
    if time_dt_ms is not None:
        time_kwargs["dt"] = time_dt_ms / 1e3
    elif time_steps is not None:
        time_kwargs["num_steps"] = time_steps

    specs = [
        PlotSpec(
            name="transition_heatmap",
            manipulation="transitions",
            metric="tpm_heatmap",
            plot_type="heatmap",
            title="Transition Probability Matrix",
            xlabel="To (attractor idx)",
            ylabel="From (attractor idx)",
            plot_kwargs={
                "cmap": "magma",
                "log_transform": True,
                "log_eps": 1e-6,
                "colorbar_label": "log10(prob + 1e-6)",
            },
        ),
    ]

    seaborn_specs = [
        PlotSpec(
            name="duration_hist",
            metric="02_occurrence_duration_hist",
            plot_type="hist",
            title="Occurrence Duration Distribution",
            xlabel="Duration (ms)",
            ylabel="Count",
            plot_kwargs={"bins": 40, "alpha": 0.7, "color": COLORS["primary"]},
        ),
        PlotSpec(
            name="lifespan_scatter",
            manipulation="per_attractor",
            metric="01_occurrences_vs_lifespan_scatter",
            hue="labels",
            size="std_duration",
            plot_type="scatter",
            title="Mean Attractor Lifespan vs Occurrences",
            xlabel="Occurrences",
            ylabel="Mean lifespan (ms)",
            plot_kwargs={
                "alpha": 0.5,
                "palette": "tab10",
                "sizes": (20, 160),
                "legend_mode": "hue",
                "xscale": "log",
            },
        ),
        PlotSpec(
            name="size_vs_occurrence_violin",
            manipulation="per_attractor",
            metric="03_size_vs_occurrence_violin",
            plot_type="violin",
            title="Attractor Size vs Occurrences",
            xlabel="Attractor size (cluster count)",
            ylabel="Occurrences",
            plot_kwargs={"alpha": 0.6, "s": 18, "color": COLORS["primary"]},
        ),
        PlotSpec(
            name="size_vs_lifespan_violin",
            manipulation="per_attractor",
            metric="04_size_vs_lifespan_violin",
            plot_type="violin",
            title="Attractor Size vs Mean Lifespan",
            xlabel="Attractor size (cluster count)",
            ylabel="Mean lifespan (ms)",
            plot_kwargs={"alpha": 0.6, "s": 18, "color": COLORS["accent"]},
        ),
        PlotSpec(
            name="lifespan_distribution_hist",
            manipulation="per_attractor",
            metric="05_lifespan_distribution_hist",
            plot_type="hist",
            title="Mean Lifespan Distribution (per attractor)",
            xlabel="Mean lifespan (ms)",
            ylabel="Count",
            plot_kwargs={"bins": 40, "alpha": 0.7, "color": COLORS["scatter"]},
        ),
        PlotSpec(
            name="mean_lifespan_by_size_bar",
            manipulation="num_clusters",
            metric="06_mean_lifespan_by_size_bar",
            plot_type="bar",
            title="Mean Lifespan by Attractor Size",
            xlabel="Attractor size (cluster count)",
            ylabel="Mean lifespan (ms)",
            plot_kwargs={"color": COLORS["tertiary"], "alpha": 0.7},
        ),
        PlotSpec(
            name="discovery_by_num_clusters_line",
            manipulation="per_attractor",
            metric="07_discovery_by_num_clusters_line",
            plot_type="line",
            hue="label",
            title="Discovery by Number of Clusters",
            xlabel="time / total sim duration",
            ylabel="# attractors discovered as share of total",
            plot_kwargs={"alpha": 0.5, "linewidth": 5.0, "palette": "tab10"},
        ),
        PlotSpec(
            name="discovery_rate_line",
            manipulation="time_evolution",
            metric="08_discovery_rate_line",
            plot_type="line",
            title="Attractor Discovery Rate",
            xlabel="Time (min)",
            ylabel="New attractors per second",
            manipulation_kwargs=time_kwargs,
            metric_kwargs={"time_unit": "min"},
            plot_kwargs={"color": COLORS["primary"], "linewidth": 5, "alpha": 0.5},
        ),
        PlotSpec(
            name="cumulative_attractors",
            manipulation="time_evolution",
            metric="09_cumulative_attractors_line",
            plot_type="line",
            title="Cumulative Attractors (per second)",
            xlabel="Time (min)",
            ylabel="Attractors per second",
            manipulation_kwargs=time_kwargs,
            metric_kwargs={"time_unit": "min"},
            plot_kwargs={"color": COLORS["secondary"], "linewidth": 5, "alpha": 0.5},
        ),
        PlotSpec(
            name="transition_l2_norm_line",
            manipulation="time_evolution",
            metric="10_tpm_L2_norm_vs_time_line",
            plot_type="line",
            title="Transition Matrix L2 Norm Over Time",
            xlabel="Time (min)",
            ylabel="L2 norm",
            manipulation_kwargs=time_kwargs,
            metric_kwargs={"time_unit": "min"},
            plot_kwargs={"color": COLORS["tertiary"], "linewidth": 5, "alpha": 0.5},
        ),
    ]

    return ComposablePlotter([
        SpecPlotter(specs),
        SeabornPlotter(specs=seaborn_specs, apply_journal_style=True),
    ])


def load_seeds_from_file(seeds_file: Path) -> list[int]:
    """Load seeds from a text file (one seed per line)."""
    with open(seeds_file) as f:
        return [int(line.strip()) for line in f if line.strip()]


def load_existing_data(save_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load existing spike data files from save_dir/data.

    Returns:
        Tuple of (raw_outputs, metadata) where each raw_output contains
        spikes_path and clusters_path.
    """
    data_dir = save_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all spike files (repeat_N_spikes.npz pattern)
    spike_files = sorted(data_dir.glob("repeat_*_spikes.npz"))
    if not spike_files:
        # Try .npy format
        spike_files = sorted(data_dir.glob("repeat_*_spikes.npy"))

    if not spike_files:
        raise FileNotFoundError(f"No spike files found in {data_dir}")

    raw_outputs = []
    metadata = []

    for spike_file in spike_files:
        # Extract repeat index from filename (repeat_0_spikes.npz -> 0)
        name = spike_file.stem  # repeat_0_spikes
        parts = name.split("_")
        repeat_idx = int(parts[1])

        raw_outputs.append({
            "spikes_path": str(spike_file),
            "clusters_path": None,  # clusters are embedded in npz
        })
        metadata.append({
            "seed": None,  # Unknown from files
            "repeat_idx": repeat_idx,
        })

    print(f"Found {len(raw_outputs)} existing data files in {data_dir}")
    return raw_outputs, metadata


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    args = _build_parser(root).parse_args()

    _apply_style(args.style, root)

    config_path = resolve_path(root, args.config)
    save_dir = resolve_path(root, args.save_dir)

    # Save command for reproducibility
    save_cmd(save_dir / "metadata")
    config_path.copy(save_dir / "metadata/config.yaml")

    # Load explicit seeds if provided
    seeds = None
    if args.seeds_file:
        seeds_file = resolve_path(root, args.seeds_file)
        if seeds_file.exists():
            seeds = load_seeds_from_file(seeds_file)
            print(f"Loaded {len(seeds)} seeds from {seeds_file}")

    # Load config to extract n_excitatory_clusters
    with open(config_path) as f:
        sim_config = yaml.safe_load(f)
    n_excitatory_clusters = sim_config.get("architecture", {}).get("clusters", {}).get("n_clusters")

    # Create pipeline components
    simulator_factory = create_simulator_factory(
        config_path,
        raster_plots=args.raster_plots,
        save_dir=save_dir,
    )
    processor_factory = create_processor_factory()
    batch_processor_factory = SNNBatchProcessorFactory(
        clustering_params={"n_excitatory_clusters": n_excitatory_clusters},
    )
    plotter = None if args.no_plots else create_plotter(
        time_dt_ms=args.time_dt_ms,
        time_steps=args.time_steps,
    )

    # Create pipeline
    pipeline = Pipeline(
        simulator_factory=simulator_factory,
        processor_factory=processor_factory,
        batch_processor_factory=batch_processor_factory,
        analyzer_factory=_ExpSNNAnalyzer,
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
        save_raw=False,  # Don't accumulate raw outputs in memory
        save_processed=True,
        save_analysis=True,
        save_plots=not args.no_plots,
        log_level=args.log_level,
        verbose_memory=args.verbose_memory,
        time_evolution_dt=(
            args.time_dt_ms / 1e3 if args.time_dt_ms is not None else None
        ),
        time_evolution_num_steps=args.time_steps,
    )

    # Process-only mode: skip simulation, load existing data
    if args.process_only:
        print("Process-only mode: loading existing data files...")
        raw_outputs, metadata = load_existing_data(save_dir)
        result = pipeline.process_existing(config, raw_outputs, metadata)
    else:
        # Run full pipeline (simulation + processing)
        result = pipeline.run(config)

    # Merge raster plots into a single PDF
    # if args.raster_plots:
    #     raster_dir = save_dir / "plots" / "rasters"
    #     if raster_dir.is_dir() and any(raster_dir.glob("*.png")):
    #         try:
    #             folder_plots_to_pdf(
    #                 raster_dir,
    #                 output_path=save_dir / "plots" / "rasters.pdf",
    #             )
    #         except ValueError as exc:
    #             print(f"Skipping raster PDF export: {exc}")

    # Merge all plots into a single PDF
    plots_dir = save_dir / "plots"
    if not args.no_plots and plots_dir.is_dir() and any(plots_dir.glob("*.png")):
        try:
            folder_plots_to_pdf(
                plots_dir,
                output_path=plots_dir / "analysis_report.pdf",
            )
            print(f"Analysis report saved to: {plots_dir / 'analysis_report.pdf'}")
        except ValueError as exc:
            print(f"Skipping PDF export: {exc}")

    # Print summary
    print("\n" + "=" * 60)
    print("SNN SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Seeds used: {len(result.seeds_used)} seeds")
    print(f"Results saved to: {save_dir}")

    if "aggregated" in result.dataframes:
        df = result.dataframes["aggregated"]
        print(f"Total occurrences: {len(df)}")
        print(f"Unique attractors: {df['attractor_idx'].nunique()}")
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
