#!/usr/bin/env python
"""Run perturbed SNN simulations using the Pipeline architecture.

This script runs SNN simulations with vectorial perturbations,
using the Pipeline for reproducibility, processing, and analysis.

Usage:
    python scripts/run_perturbed_snn.py --config configs/snn_params_with_perturbation.yaml

    # Multiple repeats with unified processing
    python scripts/run_perturbed_snn.py --config configs/snn_params_with_perturbation.yaml --n-repeats 5 --unified
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import yaml

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
from neuro_mod.execution.helpers import resolve_path, save_cmd
from neuro_mod.execution.helpers.logger import Logger
from neuro_mod.core.perturbations.vectorial import VectorialPerturbation


def _build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run perturbed SNN simulations with the Pipeline architecture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        default=str(root / "configs/snn_params_with_perturbation.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(root / "simulations/perturbed_snn"),
        help="Output directory for simulation artifacts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=256,
        help="Base seed for reproducibility.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Number of repeats to run.",
    )
    parser.add_argument(
        "--seeds-file",
        default=None,
        help="Optional path to a seeds.txt file.",
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
        help="Use unified processing for consistent attractor IDs across repeats.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level.",
    )
    return parser


def _build_perturbator(config: dict, name: str) -> VectorialPerturbation:
    """Build a VectorialPerturbation from config for a target parameter.

    Args:
        config: Full YAML configuration dictionary.
        name: Perturbation target name under the config.

    Returns:
        Configured VectorialPerturbation instance.
    """
    perturbation = dict(config.get("perturbation", {}).get(name, {}))
    vectors = perturbation.pop("vectors", [])
    perturbation.pop("params", None)
    perturbation.pop("time_dependence", None)
    seed = perturbation.pop("seed", 256)
    length = config["architecture"]["clusters"]["total_pops"]
    params = {
        **perturbation,
        "rng": np.random.default_rng(seed),
        "length": length,
    }
    return VectorialPerturbation(*vectors, **params)


def _get_time_vector(config: dict, name: str) -> np.ndarray | None:
    """Build a time mask vector for a named perturbation config.

    Args:
        config: Full YAML configuration dictionary.
        name: Perturbation target name under the config.

    Returns:
        Time mask vector or None if not configured.
    """
    perturbation = config.get("perturbation", {}).get(name, {})
    time_dependence = perturbation.get("time_dependence")
    if not time_dependence or "shape" not in time_dependence:
        return None
    dt = config["init_params"]["delta_t"]
    duration = config["init_params"]["duration_sec"]
    n_steps = int(duration // dt)
    time_vec = np.zeros(n_steps)
    onset = int(time_dependence["onset_time"] // dt)
    offset = time_dependence.get("offset_time")
    offset = offset if offset is None else int(offset // dt)
    time_vec[slice(onset, offset)] = 1
    return time_vec


def _generate_perturbations(config: dict, logger: Logger | None = None) -> dict:
    """Generate perturbations for all configured targets.

    Args:
        config: Full YAML configuration dictionary.
        logger: Optional logger for summary statistics.

    Returns:
        Dictionary of perturbation arrays keyed by target name.
    """
    perturbations = {}
    for name, cfg in config.get("perturbation", {}).items():
        if not isinstance(cfg, dict) or "params" not in cfg:
            continue
        perturbator = _build_perturbator(config, name)
        coeffs = np.asarray(cfg["params"], dtype=float)
        time_vec = _get_time_vector(config, name)
        if time_vec is not None:
            coeffs = np.outer(coeffs, time_vec)
        values = perturbator.get_perturbation(*coeffs)
        perturbations[name] = values
        if logger is not None:
            arr = np.asarray(values, dtype=float)
            logger.info(
                f"Perturbation {name}: shape={arr.shape} "
                f"mean={arr.mean():.4f} min={arr.min():.4f} max={arr.max():.4f}"
            )
    return perturbations


def create_perturbed_simulator_factory(config: dict, perturbations: dict):
    """Create a factory that produces perturbed SNN simulators."""

    def factory(seed: int, **kwargs) -> StageSNNSimulation:
        # Create stager with perturbations injected
        stager = StageSNNSimulation.from_config_dict(
            config,
            random_seed=seed,
            perturbations=perturbations,
        )
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
    """Create a SeabornPlotter with perturbation-relevant plot specifications."""
    specs = [
        PlotSpec(
            name="duration_distribution",
            plot_type="hist",
            x="duration_ms",
            title="Attractor Duration Distribution (Perturbed)",
            xlabel="Duration (ms)",
            ylabel="Count",
            kwargs={"bins": 40, "kde": True, "color": "#e76f51"},
        ),
        PlotSpec(
            name="duration_over_time",
            plot_type="scatter",
            x="start",
            y="duration_ms",
            title="Attractor Duration Over Time",
            xlabel="Start Time (s)",
            ylabel="Duration (ms)",
            kwargs={"alpha": 0.3, "s": 10, "color": "#2a9d8f"},
        ),
    ]
    return SeabornPlotter(specs=specs, apply_journal_style=True)


def load_seeds_from_file(seeds_file: Path) -> list[int]:
    """Load seeds from a text file."""
    with open(seeds_file) as f:
        return [int(line.strip()) for line in f if line.strip()]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    args = _build_parser(root).parse_args()

    config_path = resolve_path(root, args.config)
    save_dir = resolve_path(root, args.save_dir)

    # Save command for reproducibility
    save_cmd(save_dir / "metadata")

    logger = Logger(name="PerturbedSNN")

    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Generate perturbations
    logger.info("Generating perturbations from config")
    perturbations = _generate_perturbations(config, logger=logger)

    # Save perturbations for reproducibility
    perturbations_dir = save_dir / "metadata"
    perturbations_dir.mkdir(parents=True, exist_ok=True)
    np.savez(perturbations_dir / "perturbations.npz", **perturbations)
    logger.info(f"Saved perturbations to {perturbations_dir / 'perturbations.npz'}")

    # Load explicit seeds if provided
    seeds = None
    if args.seeds_file:
        seeds_file = resolve_path(root, args.seeds_file)
        if seeds_file.exists():
            seeds = load_seeds_from_file(seeds_file)
            logger.info(f"Loaded {len(seeds)} seeds from {seeds_file}")

    # Determine execution mode
    mode = ExecutionMode.REPEATED if args.n_repeats > 1 else ExecutionMode.SINGLE

    # Create pipeline components
    simulator_factory = create_perturbed_simulator_factory(config, perturbations)
    processor_factory = create_processor_factory()
    batch_processor_factory = (
        SNNBatchProcessorFactory() if args.unified and args.n_repeats > 1 else None
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
    pipeline_config = PipelineConfig(
        mode=mode,
        n_repeats=args.n_repeats,
        base_seed=args.seed,
        seeds=seeds,
        parallel=args.parallel,
        max_workers=args.max_workers,
        executor=args.executor,
        save_dir=save_dir,
        save_processed=True,
        save_analysis=True,
        save_plots=not args.no_plots,
        unified_processing=args.unified,
        log_level=args.log_level,
    )

    # Run pipeline
    logger.info("Starting perturbed SNN pipeline")
    result = pipeline.run(pipeline_config)

    # Print summary
    print("\n" + "=" * 60)
    print("PERTURBED SNN SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Seeds used: {result.seeds_used}")
    print(f"Results saved to: {save_dir}")

    # Show perturbation info
    print("\nPerturbations applied:")
    for name, values in perturbations.items():
        arr = np.asarray(values, dtype=float)
        print(f"  {name}: shape={arr.shape}, mean={arr.mean():.4f}")

    # Show metrics
    result_key = "aggregated" if mode == ExecutionMode.REPEATED else "single"
    if result_key in result.metrics:
        metrics = result.metrics[result_key]
        print(f"\n{result_key.title()} Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
