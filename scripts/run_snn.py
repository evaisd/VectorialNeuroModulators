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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from neuro_mod.pipeline import (
    ExecutionMode,
    Pipeline,
    PipelineConfig,
    PipelineResult,
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
from neuro_mod.visualization import folder_plots_to_pdf, journal_style


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


# =============================================================================
# Color Palette
# =============================================================================

COLORS = {
    "primary": "#2a9d8f",
    "secondary": "#e76f51",
    "tertiary": "#264653",
    "accent": "#f4a261",
    "neutral": "#6c757d",
    "bar": "#2a6f97",
    "scatter": "#8d99ae",
}


# =============================================================================
# Seaborn-based Plots (occurrence-level)
# =============================================================================


def create_occurrence_plotter() -> SeabornPlotter:
    """Create a SeabornPlotter for occurrence-level plots."""
    specs = [
        PlotSpec(
            name="01_duration_distribution",
            plot_type="hist",
            x="duration",
            title="Occurrence Duration Distribution",
            xlabel="Duration (ms)",
            ylabel="Count",
            kwargs={"bins": 40, "kde": True, "color": COLORS["primary"]},
        ),
        PlotSpec(
            name="02_duration_over_time",
            plot_type="scatter",
            x="t_start",
            y="duration",
            hue="num_clusters",
            title="Occurrence Duration Over Time",
            xlabel="Start Time (s)",
            ylabel="Duration (ms)",
            kwargs={"alpha": 0.4, "s": 12, "palette": "viridis"},
        ),
    ]
    return SeabornPlotter(specs=specs, apply_journal_style=True)


# =============================================================================
# Custom Matplotlib Plots (aggregated/analyzer-level)
# =============================================================================


def _aggregate_per_attractor(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate occurrence DataFrame to per-attractor summary."""
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby("attractor_idx").agg(
        occurrences=("idx", "count"),
        total_duration=("duration", "sum"),
        mean_duration=("duration", "mean"),
        std_duration=("duration", "std"),
        num_clusters=("num_clusters", "first"),
        first_start=("t_start", "min"),
        last_end=("t_end", "max"),
    ).reset_index()
    agg["std_duration"] = agg["std_duration"].fillna(0)
    return agg


def generate_aggregated_plots(data: pd.DataFrame, save_dir: Path) -> None:
    """Generate all aggregated attractor plots with proper naming."""
    if data.empty:
        return

    journal_style.apply_journal_style()
    agg = _aggregate_per_attractor(data)
    if agg.empty:
        return

    # 03: Lifespan vs Occurrences (scatter, log x, colored by cluster size)
    fig, ax = plt.subplots(figsize=(8, 5))
    unique_sizes = sorted(agg["num_clusters"].unique())
    cmap = plt.get_cmap("tab20", max(len(unique_sizes), 1))
    for i, size in enumerate(unique_sizes):
        mask = agg["num_clusters"] == size
        ax.scatter(
            agg.loc[mask, "occurrences"],
            agg.loc[mask, "mean_duration"],
            s=18,
            color=cmap(i),
            alpha=0.7,
            label=str(size),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Occurrences (log scale)")
    ax.set_ylabel("Mean lifespan (ms)")
    ax.set_title("Mean Attractor Lifespan vs Occurrences")
    ax.legend(
        title="Clusters",
        ncol=1,
        frameon=False,
        fontsize=7,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
    )
    fig.savefig(save_dir / "03_lifespan_vs_occurrences.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 04: Size Correlations (2-panel)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sizes = agg["num_clusters"].values
    occurrences = agg["occurrences"].values
    mean_lifespan = agg["mean_duration"].values
    unique_sizes_arr = np.unique(sizes)
    mean_occ_by_size = [np.mean(occurrences[sizes == s]) for s in unique_sizes_arr]
    mean_life_by_size = [np.mean(mean_lifespan[sizes == s]) for s in unique_sizes_arr]

    axes[0].scatter(sizes, occurrences, s=16, alpha=0.4, color=COLORS["primary"])
    axes[0].plot(unique_sizes_arr, mean_occ_by_size, color="#1f6f5b", linewidth=2, marker="o", markersize=4)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Attractor size (cluster count)")
    axes[0].set_ylabel("Occurrences (log scale)")
    axes[0].set_title("Size vs Occurrences")

    axes[1].scatter(sizes, mean_lifespan, s=16, alpha=0.4, color=COLORS["accent"])
    axes[1].plot(unique_sizes_arr, mean_life_by_size, color="#c26d3b", linewidth=2, marker="o", markersize=4)
    axes[1].set_xlabel("Attractor size (cluster count)")
    axes[1].set_ylabel("Mean lifespan (ms)")
    axes[1].set_title("Size vs Mean Lifespan")

    fig.suptitle("Attractor Size Correlations", fontsize=12, fontweight="bold")
    fig.savefig(save_dir / "04_size_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 05: Mean Lifespan Histogram (per attractor)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(
        agg["mean_duration"],
        bins=40,
        kde=True,
        color=COLORS["scatter"],
        edgecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Mean lifespan (ms)")
    ax.set_ylabel("Count")
    ax.set_title("Mean Lifespan Distribution (per attractor)")
    fig.savefig(save_dir / "05_mean_lifespan_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 06: Mean Lifespan by Size (bar)
    fig, ax = plt.subplots(figsize=(7, 5))
    size_summary = agg.groupby("num_clusters").agg(
        mean_lifespan=("mean_duration", "mean"),
        sem_lifespan=("mean_duration", "sem"),
    ).reset_index()
    ax.bar(
        size_summary["num_clusters"],
        size_summary["mean_lifespan"],
        yerr=size_summary["sem_lifespan"],
        color=COLORS["tertiary"],
        capsize=3,
        edgecolor="white",
    )
    ax.set_xlabel("Attractor size (cluster count)")
    ax.set_ylabel("Mean lifespan (ms)")
    ax.set_title("Mean Lifespan by Attractor Size")
    fig.savefig(save_dir / "06_mean_lifespan_by_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 07: Top Attractors (bar)
    fig, ax = plt.subplots(figsize=(10, 5))
    top = agg.nlargest(20, "total_duration")
    labels = [f"A{idx}" for idx in top["attractor_idx"]]
    ax.bar(labels, top["total_duration"], color=COLORS["bar"], edgecolor="white")
    ax.set_ylabel("Total duration (ms)")
    ax.set_title("Top Attractors by Total Occupancy")
    ax.tick_params(axis="x", rotation=45)
    fig.savefig(save_dir / "07_top_attractors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 08: Transition Heatmap
    if "prev_attractor_idx" in data.columns:
        df = data.dropna(subset=["prev_attractor_idx"]).copy()
        if not df.empty:
            df["prev_attractor_idx"] = df["prev_attractor_idx"].astype(int)
            top_n = 30
            top_attractors = data["attractor_idx"].value_counts().head(top_n).index.tolist()
            df_top = df[
                df["attractor_idx"].isin(top_attractors) &
                df["prev_attractor_idx"].isin(top_attractors)
            ]

            if not df_top.empty:
                fig, ax = plt.subplots(figsize=(8, 7))
                trans_counts = pd.crosstab(
                    df_top["prev_attractor_idx"],
                    df_top["attractor_idx"],
                    dropna=False,
                )
                row_sums = trans_counts.sum(axis=1)
                trans_probs = trans_counts.div(row_sums, axis=0).fillna(0)
                log_probs = np.log10(trans_probs.values + 1e-6)

                im = ax.imshow(log_probs, cmap="magma", aspect="auto")
                ax.set_xlabel("To")
                ax.set_ylabel("From")
                ax.set_title("Transition Matrix (Top Attractors)")
                plt.colorbar(im, ax=ax, label="log10(prob + 1e-6)")
                fig.savefig(save_dir / "08_transition_heatmap.png", dpi=150, bbox_inches="tight")
                plt.close(fig)


# =============================================================================
# Time Evolution Plots
# =============================================================================


def plot_discovery_rate(time_df: pd.DataFrame, ax, **kwargs):
    """Line plot: new attractor discovery rate over time."""
    if time_df.empty or "unique_attractors_count" not in time_df.columns:
        return

    time_s = time_df["time_ms"].values / 1000.0
    unique_counts = time_df["unique_attractors_count"].values

    # Compute discovery rate (new attractors per second)
    if len(time_s) > 1:
        dt = np.diff(time_s)
        new_attractors = np.diff(unique_counts)
        rate = new_attractors / dt
        time_centers = (time_s[:-1] + time_s[1:]) / 2

        ax.plot(time_centers, rate, color=COLORS["primary"], linewidth=1.5)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("New attractors per second")
        ax.set_title("Attractor Discovery Rate")
        ax.set_xlim(0, time_s[-1])


def plot_l2_norms(time_df: pd.DataFrame, ax, **kwargs):
    """Line plot: transition matrix L2 norms over time."""
    if time_df.empty or "transition_l2_norm" not in time_df.columns:
        return

    time_s = time_df["time_ms"].values / 1000.0
    l2_norms = time_df["transition_l2_norm"].values

    ax.plot(time_s, l2_norms, color=COLORS["secondary"], linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("L2 norm")
    ax.set_title("Transition Matrix L2 Norm Over Time")
    ax.set_xlim(0, time_s[-1])


def plot_unique_attractors(time_df: pd.DataFrame, ax, **kwargs):
    """Line plot: cumulative unique attractors over time."""
    if time_df.empty or "unique_attractors_count" not in time_df.columns:
        return

    time_s = time_df["time_ms"].values / 1000.0
    unique_counts = time_df["unique_attractors_count"].values

    ax.plot(time_s, unique_counts, color=COLORS["tertiary"], linewidth=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Unique attractors")
    ax.set_title("Cumulative Unique Attractors")
    ax.set_xlim(0, time_s[-1])


def generate_time_evolution_plots(time_df: pd.DataFrame, save_dir: Path) -> None:
    """Generate time evolution plots from the time DataFrame."""
    if time_df.empty:
        return

    journal_style.apply_journal_style()

    # Discovery rate
    fig, ax = plt.subplots(figsize=(10, 3.5))
    plot_discovery_rate(time_df, ax)
    fig.savefig(save_dir / "10_discovery_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # L2 norms
    fig, ax = plt.subplots(figsize=(10, 3.5))
    plot_l2_norms(time_df, ax)
    fig.savefig(save_dir / "11_l2_norms.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Unique attractors
    fig, ax = plt.subplots(figsize=(10, 3.5))
    plot_unique_attractors(time_df, ax)
    fig.savefig(save_dir / "12_unique_attractors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Combined Plotter Factory
# =============================================================================


def create_plotter() -> SeabornPlotter:
    """Create a plotter for occurrence-level plots (used by Pipeline)."""
    return create_occurrence_plotter()


def generate_all_plots(result: PipelineResult, save_dir: Path) -> None:
    """Generate all plots from pipeline result.

    This is called after pipeline.run() to generate:
    - Aggregated plots (per-attractor summaries, transitions)
    - Time evolution plots (discovery rate, L2 norms)

    The occurrence-level plots are handled by the Pipeline's plotter.
    """
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Get the main DataFrame
    df_key = "unified" if "unified" in result.dataframes else "aggregated"
    if df_key in result.dataframes:
        print(f"Generating aggregated plots from {df_key}...")
        generate_aggregated_plots(result.dataframes[df_key], plots_dir)

    # Get time evolution DataFrame
    time_key = "unified_time" if "unified_time" in result.dataframes else "aggregated_time"
    if time_key in result.dataframes:
        print(f"Generating time evolution plots from {time_key}...")
        generate_time_evolution_plots(result.dataframes[time_key], plots_dir)


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

    # Ensure plots directory exists
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

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

    # Generate aggregated and time evolution plots
    if not args.no_plots:
        generate_all_plots(result, save_dir)

    # Merge raster plots into a single PDF
    if args.raster_plots:
        raster_dir = plots_dir / "rasters"
        if raster_dir.is_dir():
            try:
                folder_plots_to_pdf(
                    raster_dir,
                    output_path=plots_dir / "rasters.pdf",
                )
            except ValueError as exc:
                print(f"Skipping raster PDF export: {exc}")

    # Merge all plots into a single PDF
    if not args.no_plots and plots_dir.is_dir():
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
