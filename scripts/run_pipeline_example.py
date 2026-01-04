#!/usr/bin/env python
"""Example script demonstrating the experiment Pipeline with unified processing.

This script shows how to use the Pipeline class to run experiments with
full reproducibility, logging, unified attractor identification, and
automatic plotting via seaborn.

Usage:
    # Single run
    python scripts/run_pipeline_example.py --config configs/snn_test_run.yaml

    # Repeated runs with unified processing (consistent attractor IDs)
    python scripts/run_pipeline_example.py --mode repeated --n-repeats 5 --unified

    # Parameter sweep with repeats (hierarchy: sweep_value -> attractors)
    python scripts/run_pipeline_example.py --mode sweep_repeated --n-repeats 3 --unified
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Add repo root to path if running as script
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from neuro_mod.pipeline import (
    ExecutionMode,
    Pipeline,
    PipelineConfig,
    PlotSpec,
    SeabornPlotter,
)
from neuro_mod.core.spiking_net.analysis import SNNAnalyzer
from neuro_mod.core.spiking_net.processing import (
    SNNProcessor,
    SNNBatchProcessorFactory,
)
from neuro_mod.execution.stagers import StageSNNSimulation


def create_simulator_factory(config_path: Path):
    """Create a factory that produces SNN simulators with different seeds."""

    def factory(seed: int, sweep_param=None, sweep_value=None, **kwargs):
        """Create a StageSNNSimulation instance.

        If sweep_param/sweep_value are provided, they would be used to modify
        the simulation configuration (not implemented in this example).
        """
        stager = StageSNNSimulation(config_path, random_seed=seed)
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
    """Create a SeabornPlotter with custom plot specifications."""
    specs = [
        PlotSpec(
            name="duration_distribution",
            plot_type="hist",
            x="duration",
            title="Attractor Duration Distribution",
            xlabel="Duration (ms)",
            ylabel="Count",
            kwargs={"bins": 30, "kde": True},
        ),
        PlotSpec(
            name="duration_by_repeat",
            plot_type="box",
            x="repeat",
            y="duration",
            title="Duration Distribution by Repeat",
            xlabel="Repeat",
            ylabel="Duration (ms)",
        ),
        PlotSpec(
            name="start_vs_duration",
            plot_type="scatter",
            x="t_start",
            y="duration",
            hue="repeat",
            title="Attractor Start Time vs Duration",
            xlabel="Start Time (s)",
            ylabel="Duration (ms)",
            kwargs={"alpha": 0.5, "s": 20},
        ),
    ]

    return SeabornPlotter(specs=specs, apply_journal_style=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run experiment pipeline with unified processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "snn_test_run.yaml",
        help="Path to simulation config YAML",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=REPO_ROOT / "experiments" / "pipeline_example",
        help="Directory to save results",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "repeated", "sweep", "sweep_repeated"],
        default="single",
        help="Execution mode",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Number of repeats (for repeated/sweep_repeated modes)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel execution",
    )
    parser.add_argument(
        "--unified",
        action="store_true",
        help="Use unified processing for consistent attractor IDs across repeats",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("PipelineExample")

    # Validate config exists
    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        return 1

    logger.info(f"Running pipeline example with config: {args.config}")
    logger.info(f"Mode: {args.mode}, Unified: {args.unified}, Save dir: {args.save_dir}")

    # Create pipeline components
    simulator_factory = create_simulator_factory(args.config)
    processor_factory = create_processor_factory()

    # Create batch processor factory if unified processing is enabled
    batch_processor_factory = SNNBatchProcessorFactory() if args.unified else None

    plotter = create_plotter()

    # Create pipeline
    pipeline = Pipeline(
        simulator_factory=simulator_factory,
        processor_factory=processor_factory,
        batch_processor_factory=batch_processor_factory,
        analyzer_factory=SNNAnalyzer,
        plotter=plotter,
    )

    # Determine execution mode
    mode_map = {
        "single": ExecutionMode.SINGLE,
        "repeated": ExecutionMode.REPEATED,
        "sweep": ExecutionMode.SWEEP,
        "sweep_repeated": ExecutionMode.SWEEP_REPEATED,
    }
    mode = mode_map[args.mode]

    # Build config
    config_kwargs = {
        "mode": mode,
        "base_seed": args.seed,
        "parallel": args.parallel,
        "save_dir": args.save_dir,
        "log_level": args.log_level,
        "save_processed": True,
        "save_plots": True,
        "unified_processing": args.unified,
    }

    if mode in (ExecutionMode.REPEATED, ExecutionMode.SWEEP_REPEATED):
        config_kwargs["n_repeats"] = args.n_repeats

    if mode in (ExecutionMode.SWEEP, ExecutionMode.SWEEP_REPEATED):
        # Example sweep over external input rate
        config_kwargs["sweep_param"] = ["external_currents", "nu_ext_baseline"]
        config_kwargs["sweep_values"] = [5.0, 7.5, 10.0, 12.5]

    config = PipelineConfig(**config_kwargs)

    # Run pipeline
    logger.info("Starting pipeline execution...")
    try:
        result = pipeline.run(config)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

    # Print summary
    logger.info("=" * 60)
    logger.info("PIPELINE RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Mode: {result.mode.name}")
    logger.info(f"Unified Processing: {args.unified}")
    logger.info(f"Duration: {result.duration_seconds:.2f}s")
    logger.info(f"Seeds used: {result.seeds_used}")
    logger.info(f"DataFrames: {list(result.dataframes.keys())}")
    logger.info(f"Figures generated: {len(result.figures)}")

    if "aggregated" in result.metrics:
        logger.info("Aggregated metrics:")
        for key, value in result.metrics["aggregated"].items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

    if "aggregated" in result.dataframes:
        df = result.dataframes["aggregated"]
        logger.info(f"Aggregated DataFrame shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")

        # Show metadata columns if present
        if "repeat" in df.columns:
            logger.info(f"Unique repeats: {sorted(df['repeat'].dropna().unique())}")
        if "sweep_value" in df.columns:
            logger.info(f"Unique sweep values: {sorted(df['sweep_value'].dropna().unique())}")

    logger.info("=" * 60)
    logger.info(f"Results saved to: {args.save_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
