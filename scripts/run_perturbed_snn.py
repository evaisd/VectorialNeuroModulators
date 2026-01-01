import argparse
import functools
from pathlib import Path

import numpy as np
import yaml

from neuro_mod.execution import Repeater
from neuro_mod.execution.helpers import Logger, resolve_path, save_cmd
from neuro_mod.execution.helpers.factories import make_perturbed_snn_stager
from neuro_mod.spiking_neuron_net.analysis.analyzer import Analyzer
from neuro_mod.perturbations.vectorial import VectorialPerturbation


def _build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a perturbed SNN simulation example.")
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
        "--sim-name",
        default="perturbed_arousal_long",
        help="Simulation name used under the save directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=256,
        help="Random seed for reproducibility.",
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


def main():
    root = Path(__file__).resolve().parents[1]
    args = _build_parser(root).parse_args()
    config_path = resolve_path(root, args.config)
    save_dir = resolve_path(root, args.save_dir)
    save_cmd(save_dir / "metadata")
    logger = Logger(name="PerturbedSNN")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    perturbations = _generate_perturbations(config, logger=logger)
    np.savez(save_dir / "metadata" / "perturbations.npz", **perturbations)

    repeater = Repeater(
        n_repeats=args.n_repeats,
        config=config_path,
        save_dir=save_dir,
        logger=logger,
        seed=args.seed,
        seeds_file=args.seeds_file,
        load_saved_seeds=args.load_saved_seeds,
        parallel=args.parallel,
        max_workers=args.max_workers,
        executor=args.executor,
        stager_factory=functools.partial(
            make_perturbed_snn_stager,
            config=config,
            perturbations=perturbations,
        ),
    )
    repeater.run()
    repeater.logger.info("Building analyzer and saving analysis.")
    analyzer = Analyzer(save_dir / "data", clusters=save_dir / "clusters.npy")
    analyzer.save_analysis(save_dir / "analysis")
    repeater.logger.info("Analysis generation complete.")


if __name__ == "__main__":
    main()
