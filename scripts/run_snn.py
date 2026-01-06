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
import numpy as np
import pandas as pd
import seaborn as sns

from neuro_mod.pipeline import (
    BasePlotter,
    ComposablePlotter,
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
import yaml

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


class NamedMatplotlibPlotter(BasePlotter):
    """Matplotlib plotter that saves figures with explicit names."""

    def __init__(
        self,
        plot_specs: list[tuple[str, Any, tuple[float, float] | None]],
        apply_journal_style: bool = True,
    ) -> None:
        self.plot_specs = plot_specs
        if apply_journal_style:
            try:
                from neuro_mod.visualization import journal_style
                journal_style.apply_journal_style()
            except ImportError:
                pass

    def plot(
        self,
        data: pd.DataFrame,
        metrics: dict[str, Any] | None = None,
        save_dir: Path | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        figures: list[Any] = []
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        for name, plot_fn, figsize in self.plot_specs:
            fig, ax = plt.subplots(figsize=figsize)
            maybe_fig = plot_fn(data, ax, metrics=metrics, **kwargs)
            fig_to_save = maybe_fig if hasattr(maybe_fig, "savefig") else fig
            if fig_to_save is not fig:
                plt.close(fig)
            figures.append(fig_to_save)
            if save_dir:
                fig_to_save.savefig(save_dir / f"{name}.png", dpi=150, bbox_inches="tight")
                plt.close(fig_to_save)
        return figures


def _get_per_attractor_df(
    data: pd.DataFrame,
    per_attractor_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if per_attractor_df is not None and not per_attractor_df.empty:
        return per_attractor_df
    return SNNAnalyzer.aggregate_per_attractor_df(data)


def plot_lifespan_vs_occurrences(
    data: pd.DataFrame,
    ax,
    **kwargs: Any,
) -> None:
    agg = _get_per_attractor_df(data, kwargs.get("per_attractor_df"))
    if agg.empty:
        return
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


def plot_size_correlations(
    data: pd.DataFrame,
    ax,
    **kwargs: Any,
):
    agg = _get_per_attractor_df(data, kwargs.get("per_attractor_df"))
    if agg.empty:
        return None
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
    return fig


def plot_mean_lifespan_histogram(
    data: pd.DataFrame,
    ax,
    **kwargs: Any,
) -> None:
    agg = _get_per_attractor_df(data, kwargs.get("per_attractor_df"))
    if agg.empty:
        return
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


def plot_mean_lifespan_by_size(
    data: pd.DataFrame,
    ax,
    **kwargs: Any,
) -> None:
    agg = _get_per_attractor_df(data, kwargs.get("per_attractor_df"))
    if agg.empty:
        return
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


def plot_top_attractors(
    data: pd.DataFrame,
    ax,
    **kwargs: Any,
) -> None:
    agg = _get_per_attractor_df(data, kwargs.get("per_attractor_df"))
    if agg.empty:
        return
    top = agg.nlargest(20, "total_duration")
    labels = [f"A{idx}" for idx in top["attractor_idx"]]
    ax.bar(labels, top["total_duration"], color=COLORS["bar"], edgecolor="white")
    ax.set_ylabel("Total duration (ms)")
    ax.set_title("Top Attractors by Total Occupancy")
    ax.tick_params(axis="x", rotation=45)


def plot_transition_heatmap(
    data: pd.DataFrame,
    ax,
    **kwargs: Any,
) -> None:
    if "prev_attractor_idx" not in data.columns:
        return
    df = data.dropna(subset=["prev_attractor_idx"]).copy()
    if df.empty:
        return
    df["prev_attractor_idx"] = df["prev_attractor_idx"].astype(int)
    top_n = 30
    top_attractors = data["attractor_idx"].value_counts().head(top_n).index.tolist()
    df_top = df[
        df["attractor_idx"].isin(top_attractors) &
        df["prev_attractor_idx"].isin(top_attractors)
    ]

    if df_top.empty:
        return
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


def create_aggregated_plotter() -> NamedMatplotlibPlotter:
    """Create a plotter for aggregated attractor plots."""
    return NamedMatplotlibPlotter(
        plot_specs=[
            ("03_lifespan_vs_occurrences", plot_lifespan_vs_occurrences, (8, 5)),
            ("04_size_correlations", plot_size_correlations, None),
            ("05_mean_lifespan_histogram", plot_mean_lifespan_histogram, (7, 5)),
            ("06_mean_lifespan_by_size", plot_mean_lifespan_by_size, (7, 5)),
            ("07_top_attractors", plot_top_attractors, (10, 5)),
            ("08_transition_heatmap", plot_transition_heatmap, (8, 7)),
        ],
        apply_journal_style=True,
    )


class TimeEvolutionPlotter(BasePlotter):
    """Plot time evolution data when provided via kwargs."""

    def __init__(self) -> None:
        self._plotter = SeabornPlotter(
            specs=[
                PlotSpec(
                    name="10_discovery_rate",
                    plot_type="line",
                    x="time_min",
                    y="discovery_rate_per_s",
                    title="Attractor Discovery Rate",
                    xlabel="Time (min)",
                    ylabel="New attractors per second",
                    kwargs={"color": COLORS["primary"], "linewidth": 3, "alpha":.5},
                ),
                PlotSpec(
                    name="11_transition_l2_norm",
                    plot_type="line",
                    x="time_min",
                    y="transition_l2_norm",
                    title="Transition Matrix L2 Norm Over Time",
                    xlabel="Time (min)",
                    ylabel="L2 norm",
                    kwargs={"color": COLORS["secondary"], "linewidth": 3, "alpha":.5},
                ),
                PlotSpec(
                    name="12_unique_attractors",
                    plot_type="line",
                    x="time_min",
                    y="unique_attractors_count",
                    title="Cumulative Unique Attractors",
                    xlabel="Time (min)",
                    ylabel="Unique attractors",
                    kwargs={"color": COLORS["tertiary"], "linewidth": 3, "alpha":.5},
                ),
                PlotSpec(
                    name="13_l2_norm_vs_discovery_rate",
                    plot_type="scatter",
                    x="discovery_rate_per_s",
                    y="transition_l2_norm",
                    title="Transition L2 Norm vs Discovery Rate",
                    xlabel="Discovery rate (attractors/s)",
                    ylabel="L2 norm",
                    kwargs={"color": COLORS["accent"], "alpha": 0.6, "s": 20},
                ),
            ],
            apply_journal_style=True,
        )

    def plot(
        self,
        data: pd.DataFrame,
        metrics: dict[str, Any] | None = None,
        save_dir: Path | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        time_df = kwargs.get("time_df")
        if time_df is None or time_df.empty:
            return []
        # Convert time from ms to minutes
        time_df = time_df.copy()
        time_df["time_min"] = time_df["time_ms"] / 60000.0
        return self._plotter.plot(time_df, metrics=metrics, save_dir=save_dir)


def create_time_plotter() -> BasePlotter:
    """Create a plotter for time-evolution plots."""
    return TimeEvolutionPlotter()


# =============================================================================
# Combined Plotter Factory
# =============================================================================


def create_plotter() -> BasePlotter:
    """Create a combined plotter for all pipeline plots."""
    return ComposablePlotter([
        create_occurrence_plotter(),
        create_aggregated_plotter(),
        create_time_plotter(),
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
        from neuro_mod.pipeline.config import PipelineResult
        from neuro_mod.pipeline.io import get_git_commit, get_timestamp
        import time as time_module

        print("Process-only mode: loading existing data files...")
        raw_outputs, metadata = load_existing_data(save_dir)

        # Initialize result
        result = PipelineResult(
            mode=config.mode,
            config=config,
            timestamp=get_timestamp(),
            git_commit=get_git_commit(),
        )
        result.seeds_used = [m.get("seed") for m in metadata]

        start_time = time_module.time()

        # Run unified processing
        print(f"Running unified processing on {len(raw_outputs)} files...")
        batch_processor = batch_processor_factory(raw_outputs, metadata)
        processed = batch_processor.process_batch(raw_outputs, metadata)

        if config.save_processed:
            batch_processor.save(save_dir / "processed" / "unified")

        # Analysis
        analyzer = SNNAnalyzer(processed, config=batch_processor.get_config())
        df = analyzer.to_dataframe()
        result.dataframes["unified"] = df
        result.dataframes["aggregated"] = df
        result.metrics["unified"] = analyzer.get_summary_metrics()
        result.metrics["aggregated"] = result.metrics["unified"]

        per_attr_df = analyzer.get_per_attractor_dataframe()
        if not per_attr_df.empty:
            result.dataframes["unified_per_attractor"] = per_attr_df
            result.dataframes["aggregated_per_attractor"] = per_attr_df

        time_df = analyzer.get_time_evolution_dataframe(
            dt=config.time_evolution_dt,
            num_steps=config.time_evolution_num_steps,
        )
        if not time_df.empty:
            result.dataframes["unified_time"] = time_df
            result.dataframes["aggregated_time"] = time_df

        result.duration_seconds = time_module.time() - start_time

        # Generate plots
        if plotter and not args.no_plots:
            print("Generating plots...")
            plots_dir = save_dir / "plots"
            time_df = result.dataframes.get("aggregated_time")
            per_attr_df = result.dataframes.get("aggregated_per_attractor")
            figures = plotter.plot(
                data=result.dataframes["aggregated"],
                metrics=result.metrics.get("aggregated"),
                save_dir=plots_dir,
                time_df=time_df,
                per_attractor_df=per_attr_df,
            )
            result.figures.extend(figures)

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
