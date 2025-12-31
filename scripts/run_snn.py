
import argparse
from pathlib import Path

from neuro_mod.execution import Repeater
from neuro_mod.execution.helpers import Logger, resolve_path, save_cmd
from neuro_mod.execution.stagers import StageSNNSimulation
from neuro_mod.spiking_neuron_net.analysis.analyzer import Analyzer

def _build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an SNN test simulation.")
    parser.add_argument(
        "--config",
        default=str(root / "configs/snn_test_run.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(root / "simulations/snn_test_run"),
        help="Output directory for simulation artifacts.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
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
        stager_factory=lambda seed: StageSNNSimulation(
            config,
            random_seed=seed,
            logger=logger,
        ),
    )
    repeater.run()
    repeater.logger.info("Building analyzer and saving analysis.")
    analyzer = Analyzer(save_dir / "data", clusters=save_dir / "clusters.npy")
    analyzer.save_analysis(save_dir / "analysis")
    repeater.logger.info("Analysis generation complete.")


if __name__ == "__main__":
    main()


# codex resume 019b6a12-d22a-7283-b885-4739684a18b1
