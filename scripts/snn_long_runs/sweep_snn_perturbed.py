#!/usr/bin/env python3
"""Sweep rate perturbation values; run N repeats per value.

Example:
  python scripts/sweep_snn_perturbed.py \
    --config configs/snn_long_run.yaml \
    --save-dir simulations/snn_rate_sweep \
    --sweep-values -2 -1 0 1 2 \
    --n-repeats 20
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml
import matplotlib.pyplot as plt

from neuro_mod.core.spiking_net.processing import SNNBatchProcessorFactory, SNNProcessor
from neuro_mod.execution.helpers.cli import resolve_path, save_cmd
from neuro_mod.execution.stagers import StageSNNSimulation
from neuro_mod.pipeline import ExecutionMode, Pipeline, PipelineConfig
from neuro_mod.visualization import folder_plots_to_pdf

from run_snn import create_plotter, _ExpSNNAnalyzer, load_seeds_from_file


def _build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep rate perturbation values; run N repeats per value.",
    )
    parser.add_argument(
        "--config",
        default=str(root / "configs/snn_long_run.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(root / "simulations/snn_rate_sweep"),
        help="Output directory for sweep artifacts.",
    )
    parser.add_argument(
        "--style",
        default=str(root / "style/neuroips.mplstyle"),
        help="Matplotlib style name or path to a .mplstyle file.",
    )
    parser.add_argument(
        "--sweep-values",
        nargs="+",
        type=float,
        required=True,
        help="Rate perturbation values to sweep.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=25,
        help="Number of repeats per sweep value.",
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


def _format_sweep_dir(base: Path, value: float) -> Path:
    token = f"{value}".replace("-", "m").replace(".", "p")
    return base / f"rate_{token}"


def _load_n_clusters(config_path: Path) -> int:
    with open(config_path) as f:
        sim_config = yaml.safe_load(f)
    clusters_cfg = sim_config.get("architecture", {}).get("clusters", {})
    n_clusters = clusters_cfg.get("n_clusters")
    if n_clusters is None:
        n_clusters = clusters_cfg.get("total_pops")
    if n_clusters is None:
        raise ValueError("Missing architecture.clusters.n_clusters in config")
    return int(n_clusters)


def create_processor_factory(dt: float = 0.5e-3):
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


def create_sweep_simulator_factory(
    config_path: Path,
    rate_value: float,
    *,
    n_clusters: int,
    raster_plots: bool,
    save_dir: Path,
):
    rate_vec = np.full(n_clusters, float(rate_value), dtype=float)

    def factory(seed: int, **kwargs):
        stager = StageSNNSimulation(
            config_path,
            random_seed=seed,
            rate_perturbation=rate_vec,
        )
        if raster_plots:
            return _RasterPlotRunner(stager, seed, save_dir)
        return stager

    return factory


def _run_sweep_value(
    *,
    config_path: Path,
    save_dir: Path,
    rate_value: float,
    seeds: list[int] | None,
    args: argparse.Namespace,
    n_clusters: int,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    save_cmd(save_dir / "metadata")
    config_path.copy(save_dir / "metadata/config.yaml")

    simulator_factory = create_sweep_simulator_factory(
        config_path,
        rate_value,
        n_clusters=n_clusters,
        raster_plots=args.raster_plots,
        save_dir=save_dir,
    )
    processor_factory = create_processor_factory()
    batch_processor_factory = SNNBatchProcessorFactory(
        clustering_params={"n_excitatory_clusters": n_clusters},
    )
    plotter = None if args.no_plots else create_plotter(
        time_dt_ms=args.time_dt_ms,
        time_steps=args.time_steps,
    )

    pipeline = Pipeline(
        simulator_factory=simulator_factory,
        processor_factory=processor_factory,
        batch_processor_factory=batch_processor_factory,
        analyzer_factory=_ExpSNNAnalyzer,
        plotter=plotter,
    )

    config = PipelineConfig(
        mode=ExecutionMode.REPEATED,
        n_repeats=args.n_repeats,
        base_seed=args.seed,
        seeds=seeds,
        parallel=args.parallel,
        max_workers=args.max_workers,
        executor=args.executor,
        save_dir=save_dir,
        save_raw=False,
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

    result = pipeline.run(config)

    plots_dir = save_dir / "plots"
    if not args.no_plots and plots_dir.is_dir() and any(plots_dir.glob("*.png")):
        try:
            folder_plots_to_pdf(
                plots_dir,
                output_path=plots_dir / "analysis_report.pdf",
            )
        except ValueError as exc:
            print(f"Skipping PDF export: {exc}")

    if not args.keep_raw:
        raw_data_dir = save_dir / "data"
        if raw_data_dir.exists():
            shutil.rmtree(raw_data_dir)

    print(
        f"Sweep value {rate_value}: completed in {result.duration_seconds:.2f}s."
        f" Results at {save_dir}"
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    args = _build_parser(root).parse_args()

    _apply_style(args.style, root)

    config_path = resolve_path(root, args.config)
    save_root = resolve_path(root, args.save_dir)

    seeds = None
    if args.seeds_file:
        seeds_file = resolve_path(root, args.seeds_file)
        if seeds_file.exists():
            seeds = load_seeds_from_file(seeds_file)
            print(f"Loaded {len(seeds)} seeds from {seeds_file}")

    n_clusters = _load_n_clusters(config_path)

    for value in args.sweep_values:
        sweep_dir = _format_sweep_dir(save_root, value)
        _run_sweep_value(
            config_path=config_path,
            save_dir=sweep_dir,
            rate_value=value,
            seeds=seeds,
            args=args,
            n_clusters=n_clusters,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
