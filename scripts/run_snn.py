
import argparse
import functools
from pathlib import Path
import shutil

from neuro_mod.execution import Repeater
from neuro_mod.execution.helpers import Logger
from neuro_mod.execution.helpers.cli import resolve_path, save_cmd
from neuro_mod.execution.helpers.factories import make_snn_stager
from neuro_mod.core.spiking_net.processing import SNNProcessor


def _build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an SNN test simulation.")
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
        help="Seed for reproducible repeat selection.",
    )
    parser.add_argument(
        "--seeds-file",
        default=None,
        help="Optional path to a seeds.txt file.",
    )
    parser.add_argument(
        "--load-saved-seeds",
        action="store_true",
        help="Load seeds from a prior run if available.",
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
        help="Maximum number of workers to use for parallel execution.",
    )
    parser.add_argument(
        "--executor",
        choices=("thread", "process"),
        default="thread",
        help="Parallel executor backend.",
    )
    return parser


def main():
    root = Path(__file__).resolve().parents[1]
    args = _build_parser(root).parse_args()
    config = resolve_path(root, args.config)
    save_dir = resolve_path(root, args.save_dir)
    save_cmd(save_dir / "metadata")
    logger = Logger(name="Repeater")
    repeater = Repeater(
        n_repeats=args.n_repeats,
        config=config,
        save_dir=save_dir,
        logger=logger,
        seed=args.seed,
        seeds_file=args.seeds_file,
        load_saved_seeds=args.load_saved_seeds,
        parallel=args.parallel,
        max_workers=args.max_workers,
        executor=args.executor,
        stager_factory=functools.partial(make_snn_stager, config_path=config),
        export_plots_pdf=True,
        plots_pdf_path=save_dir / "rasters.pdf"
    )
    repeater.run()
    repeater.logger.info("Processing spike data and saving analysis.")
    processor = SNNProcessor(save_dir / "data", clusters_path=save_dir / "clusters.npy")
    processor.process()
    processor.save(save_dir / "analysis")
    repeater.logger.info("Processing complete.")
    repeater.logger.info("Removing data.")
    shutil.rmtree(save_dir / "data")


if __name__ == "__main__":
    main()


# codex resume 019b6a12-d22a-7283-b885-4739684a18b1
