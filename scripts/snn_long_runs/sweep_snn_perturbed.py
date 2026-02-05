#!/usr/bin/env python3
"""Sweep perturbation coefficients; run N repeats per value.

Outputs are stored under a single save_dir root, with per-sweep results keyed
as sweep_0, sweep_1, ... in dataframes/ and metrics/. A sweep-level summary
table and plots are saved under dataframes/sweep_summary.* and
plots/sweep_summary/.

Example:
  python scripts/sweep_snn_perturbed.py \
    --config configs/snn_long_run.yaml \
    --save-dir simulations/snn_rate_sweep \
    --sweep-values -2 -1 0 1 2 \
    --n-repeats 20

  python scripts/sweep_snn_perturbed.py \
    --config configs/snn_long_run_perturbed.yaml \
    --save-dir simulations/snn_rate_sweep \
    --range 0.05 0.15 5 \
    --n-repeats 10

  # Multi-vector coefficients
  python scripts/sweep_snn_perturbed.py \
    --config configs/snn_long_run_perturbed.yaml \
    --save-dir simulations/snn_rate_sweep \
    --params-grid 0.05,0.1 0.1,0.05 \
    --n-repeats 10
"""

from __future__ import annotations

import argparse
import copy
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
import matplotlib.pyplot as plt
from neuro_mod.core.spiking_net.processing import SNNBatchProcessorFactory, SNNProcessor
from neuro_mod.core.perturbations.vectorial import VectorialPerturbation
from neuro_mod.execution.helpers.cli import resolve_path, save_cmd
from neuro_mod.execution.helpers.logger import Logger
from neuro_mod.execution.stagers import StageSNNSimulation
from neuro_mod.pipeline import (
    ExecutionMode,
    Pipeline,
    PipelineConfig,
    build_sweep_summary,
    save_sweep_summary,
    plot_sweep_summary,
)
from neuro_mod.visualization import folder_plots_to_pdf, image_to_pdf

from run_snn import create_plotter, _ExpSNNAnalyzer, load_seeds_from_file


class SweepSimulatorFactory:
    """Picklable simulator factory for process-based parallel execution."""

    def __init__(
        self,
        config_path: Path,
        build_rate_perturbation: callable,
        base_config: dict[str, Any],
        *,
        raster_plots: bool,
        save_dir: Path,
        output_keys: list[str] | None,
        compile_net: bool,
        log_level: str,
        logger_name: str = "SweepSimulatorFactory",
    ) -> None:
        self.config_path = config_path
        self.build_rate_perturbation = build_rate_perturbation
        self.base_config = base_config
        self.raster_plots = raster_plots
        self.save_dir = save_dir
        self.output_keys = output_keys
        self.compile_net = compile_net
        self.log_level = log_level
        self.logger_name = logger_name

    def __call__(self, seed: int, **kwargs):
        sweep_value = kwargs.get("sweep_value")
        sweep_idx = kwargs.get("sweep_idx")
        params = _coerce_sweep_value(sweep_value)
        rate_perturbation = self.build_rate_perturbation(params)
        sweep_label = _format_sweep_label(params)
        _maybe_write_sweep_config(self.base_config, params, self.save_dir, sweep_idx, sweep_label)
        logger = Logger(name=self.logger_name, level=self.log_level)
        logger.debug(
            f"Sweep perturbation: sweep_value={sweep_value} "
            f"params={params} summary={_summarize_array(rate_perturbation)}"
        )
        output_keys = self.output_keys
        if self.raster_plots and output_keys is not None:
            output_keys = sorted(set(output_keys + ["spikes"]))
        stager = StageSNNSimulation(
            self.config_path,
            random_seed=seed,
            rate_perturbation=rate_perturbation,
            output_keys=output_keys,
            compile_net=self.compile_net,
            logger=logger,
        )
        if self.raster_plots:
            return _RasterPlotRunner(stager, seed, self.save_dir, sweep_label)
        return stager


def _build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep rate perturbation values; run N repeats per value.",
    )
    parser.add_argument(
        "--config",
        default=str(root / "configs/snn_long_run_perturbed.yaml"),
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
    sweep_group = parser.add_mutually_exclusive_group(required=True)
    sweep_group.add_argument(
        "--sweep-values",
        nargs="+",
        type=float,
        help="Coefficient values for a single perturbation vector.",
    )
    sweep_group.add_argument(
        "--params-grid",
        nargs="+",
        default=None,
        help="Comma-separated coefficient vectors for multi-vector sweeps (e.g. 0.1,0.2 0.2,0.1).",
    )
    sweep_group.add_argument(
        "--range",
        nargs=3,
        type=float,
        metavar=("LOW", "HIGH", "NUM"),
        help="Generate evenly spaced coefficients for single-vector sweeps.",
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
        "--persist-in-worker",
        action="store_true",
        help="Persist raw outputs inside worker processes (process executor only).",
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
        "--lite-output",
        action="store_true",
        help="Only return spikes/clusters from simulations (faster, lower memory).",
    )
    parser.add_argument(
        "--output-keys",
        nargs="+",
        default=None,
        help="Explicit output keys to keep (overrides --lite-output). "
             "Accepts space- or comma-separated values, e.g. spikes clusters or spikes,clusters.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep raw spike data after processing (default: delete to save space).",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Disable compression for raw spike files (faster, larger).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile on the LIF network (PyTorch 2.x only).",
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

    def __init__(
        self,
        stager: StageSNNSimulation,
        seed: int,
        save_dir: Path,
        sweep_label: str,
    ) -> None:
        self._stager = stager
        self._seed = seed
        self._save_dir = save_dir
        self._sweep_label = sweep_label

    def run(self) -> dict[str, Any]:
        outputs = self._stager.run()
        spikes = outputs.get("spikes")
        if spikes is not None and hasattr(self._stager, "_plot"):
            plot_dir = self._save_dir / "plots" / "rasters"
            plot_dir.mkdir(parents=True, exist_ok=True)
            filename = f"spikes_{self._sweep_label}_seed_{self._seed}.png"
            png_path = plot_dir / filename
            self._stager._plot(spikes, plt_path=png_path)
            pdf_path = png_path.with_suffix(".pdf")
            try:
                image_to_pdf(png_path, pdf_path)
            except Exception as exc:
                print(f"Warning: failed to create rasterized PDF {pdf_path}: {exc}")
        return outputs


def _load_sim_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _load_n_clusters(sim_config: dict[str, Any]) -> int:
    clusters_cfg = sim_config.get("architecture", {}).get("clusters", {})
    n_clusters = clusters_cfg.get("n_clusters")
    if n_clusters is None:
        n_clusters = clusters_cfg.get("total_pops")
    if n_clusters is None:
        raise ValueError("Missing architecture.clusters.n_clusters in config")
    return int(n_clusters)


def _select_rate_perturbation_cfg(
    perturbation_cfg: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if "rate" in perturbation_cfg and isinstance(perturbation_cfg["rate"], dict):
        return perturbation_cfg["rate"], ["perturbation", "rate"]
    return perturbation_cfg, ["perturbation"]


def _get_time_vector(rate_cfg: dict[str, Any], init_params: dict[str, Any]) -> np.ndarray | None:
    time_dependence = rate_cfg.get("time_dependence")
    if not time_dependence or "shape" not in time_dependence:
        return None
    dt = init_params.get("delta_t")
    duration = init_params.get("duration_sec")
    if dt is None or duration is None:
        raise ValueError("init_params.delta_t and init_params.duration_sec are required for time_dependence")
    n_steps = int(duration // dt)
    time_vec = np.zeros(n_steps)
    onset = int(time_dependence.get("onset_time", 0) // dt)
    offset = time_dependence.get("offset_time")
    offset = None if offset is None else int(offset // dt)
    time_vec[slice(onset, offset)] = 1
    return time_vec


def _build_perturbator(
    rate_cfg: dict[str, Any],
    *,
    length: int,
) -> VectorialPerturbation:
    cfg = dict(rate_cfg)
    vectors = cfg.pop("vectors", [])
    cfg.pop("params", None)
    cfg.pop("time_dependence", None)
    seed = cfg.pop("seed", 256)
    rng = np.random.default_rng(seed)
    params = {
        **cfg,
        "rng": rng,
        "length": length,
    }
    return VectorialPerturbation(*vectors, **params)


def _build_rate_perturbation_factory(
    sim_config: dict[str, Any],
    logger: Logger | None = None,
) -> tuple[callable, int]:
    perturbation_cfg = sim_config.get("perturbation")
    if not isinstance(perturbation_cfg, dict):
        raise ValueError("Config missing perturbation block for rate sweep.")
    rate_cfg, _ = _select_rate_perturbation_cfg(perturbation_cfg)
    vectors = rate_cfg.get("vectors", [])
    clusters_cfg = sim_config.get("architecture", {}).get("clusters", {})
    length = clusters_cfg.get("total_pops")
    if length is None:
        length = _load_n_clusters(sim_config) * 2 + 2
    n_params = len(rate_cfg.get("params", [])) or len(vectors) or 1
    perturbator = _build_perturbator(rate_cfg, length=length)
    init_params = sim_config.get("init_params", {})
    time_vec = _get_time_vector(rate_cfg, init_params)
    if logger is not None:
        logger.debug(
            "Rate perturbation factory (target=rate): "
            f"n_params={n_params} vectors={len(vectors)} "
            f"length={length} time_dependence={time_vec is not None}"
        )

    def build_rate_perturbation(params: Iterable[float]) -> np.ndarray:
        coeffs = np.asarray(list(params), dtype=float)
        if coeffs.size != n_params:
            raise ValueError(f"Expected {n_params} coefficients, got {coeffs.size}.")
        if time_vec is not None:
            coeffs = np.outer(coeffs, time_vec)
        return perturbator.get_perturbation(*coeffs)

    return build_rate_perturbation, n_params


def _parse_params_grid(items: list[str] | None) -> list[list[float]]:
    if not items:
        return []
    grid: list[list[float]] = []
    for item in items:
        parts = [p for p in item.split(",") if p != ""]
        if not parts:
            continue
        grid.append([float(p) for p in parts])
    return grid


def _parse_output_keys(items: list[str] | None) -> list[str] | None:
    if not items:
        return None
    keys: list[str] = []
    for item in items:
        parts = [p.strip() for p in item.split(",") if p.strip()]
        if parts:
            keys.extend(parts)
        else:
            keys.append(item)
    seen: set[str] = set()
    deduped: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped or None


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
    build_rate_perturbation: callable,
    base_config: dict[str, Any],
    *,
    raster_plots: bool,
    save_dir: Path,
    output_keys: list[str] | None,
    compile_net: bool,
    log_level: str,
):
    return SweepSimulatorFactory(
        config_path,
        build_rate_perturbation,
        base_config,
        raster_plots=raster_plots,
        save_dir=save_dir,
        output_keys=output_keys,
        compile_net=compile_net,
        log_level=log_level,
    )


def _coerce_sweep_value(value: Any) -> list[float]:
    if value is None:
        raise ValueError("Missing sweep_value for sweep execution.")
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    return [float(value)]


def _format_sweep_label(params: list[float]) -> str:
    return "sweep_" + "_".join(f"{value:g}" for value in params)


def _summarize_array(value: Any) -> str:
    if value is None:
        return "None"
    arr = np.asarray(value, dtype=float)
    shape = arr.shape
    dtype = arr.dtype
    if arr.size == 0:
        return f"shape={shape} dtype={dtype} empty"
    return (
        "shape={shape} dtype={dtype} min={min_val:.6g} mean={mean_val:.6g} "
        "max={max_val:.6g} std={std_val:.6g}"
    ).format(
        shape=shape,
        dtype=dtype,
        min_val=float(np.min(arr)),
        mean_val=float(np.mean(arr)),
        max_val=float(np.max(arr)),
        std_val=float(np.std(arr)),
    )


def _write_sweep_config(
    base_config: dict[str, Any],
    params: list[float],
    out_path: Path,
) -> None:
    updated = copy.deepcopy(base_config)
    perturbation_cfg = updated.get("perturbation")
    if not isinstance(perturbation_cfg, dict):
        raise ValueError("Config missing perturbation block for rate sweep.")
    rate_cfg, key_path = _select_rate_perturbation_cfg(perturbation_cfg)
    rate_cfg["params"] = [float(value) for value in params]
    if key_path == ["perturbation", "rate"]:
        updated["perturbation"]["rate"] = rate_cfg
    else:
        updated["perturbation"] = rate_cfg
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(updated, f, sort_keys=False)


def _maybe_write_sweep_config(
    base_config: dict[str, Any],
    params: list[float],
    save_dir: Path,
    sweep_idx: int | None,
    sweep_label: str,
) -> None:
    if save_dir is None:
        return
    if sweep_idx is not None:
        sweep_key = f"sweep_{sweep_idx}"
    else:
        sweep_key = sweep_label
    out_path = save_dir / "metadata" / sweep_key / "config.yaml"
    if out_path.exists():
        return
    _write_sweep_config(base_config, params, out_path)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
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

    sim_config = _load_sim_config(config_path)
    n_clusters = _load_n_clusters(sim_config)
    sweep_logger = Logger(name="SweepPerturbation", level=args.log_level)
    build_rate_perturbation, n_params = _build_rate_perturbation_factory(
        sim_config,
        logger=sweep_logger,
    )
    output_keys = _parse_output_keys(args.output_keys)
    if output_keys is None and args.lite_output:
        output_keys = ["spikes", "clusters"]
    params_grid = _parse_params_grid(args.params_grid)
    if params_grid:
        if any(len(params) != n_params for params in params_grid):
            raise SystemExit(f"Each --params-grid entry must have {n_params} values.")
        sweep_params = params_grid
    elif args.range is not None:
        if n_params != 1:
            raise SystemExit(
                f"Config expects {n_params} coefficients; use --params-grid."
            )
        low, high, num = args.range
        num_int = int(num)
        if num_int <= 0:
            raise SystemExit("Range NUM must be a positive integer.")
        sweep_params = [[value] for value in np.linspace(low, high, num_int)]
    else:
        if n_params != 1:
            raise SystemExit(
                f"Config expects {n_params} coefficients; use --params-grid."
            )
        sweep_params = [[value] for value in args.sweep_values]

    save_root.mkdir(parents=True, exist_ok=True)
    save_cmd(save_root / "metadata")

    simulator_factory = create_sweep_simulator_factory(
        config_path,
        build_rate_perturbation,
        sim_config,
        raster_plots=args.raster_plots,
        save_dir=save_root,
        output_keys=output_keys,
        compile_net=args.compile,
        log_level=args.log_level,
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
        mode=ExecutionMode.SWEEP_REPEATED,
        n_repeats=args.n_repeats,
        base_seed=args.seed,
        seeds=seeds,
        sweep_param="perturbation.rate.params",
        sweep_values=sweep_params,
        parallel=args.parallel,
        max_workers=args.max_workers,
        executor=args.executor,
        persist_raw_in_worker=args.persist_in_worker,
        save_dir=save_root,
        save_raw=False,
        save_processed=True,
        save_analysis=True,
        save_plots=not args.no_plots,
        save_compressed=not args.no_compress,
        log_level=args.log_level,
        verbose_memory=args.verbose_memory,
        time_evolution_dt=(
            args.time_dt_ms / 1e3 if args.time_dt_ms is not None else None
        ),
        time_evolution_num_steps=args.time_steps,
    )

    result = pipeline.run(config)

    summary_df = build_sweep_summary(result, sweep_params)
    save_sweep_summary(save_root, summary_df)

    if not args.no_plots:
        aggregated_df = result.dataframes.get("aggregated")
        plot_sweep_summary(save_root, summary_df, aggregated_df)

        plots_dir = save_root / "plots"
        if plots_dir.is_dir():
            for subdir in sorted(p for p in plots_dir.iterdir() if p.is_dir()):
                pngs = list(subdir.glob("*.png"))
                if not pngs:
                    continue
                try:
                    folder_plots_to_pdf(
                        subdir,
                        output_path=subdir / "analysis_report.pdf",
                    )
                except ValueError as exc:
                    print(f"Skipping PDF export in {subdir}: {exc}")

    if not args.keep_raw:
        raw_data_dir = save_root / "data"
        if raw_data_dir.exists():
            shutil.rmtree(raw_data_dir)

    print(
        f"Sweep completed in {result.duration_seconds:.2f}s."
        f" Results at {save_root}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
