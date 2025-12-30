
import argparse
from pathlib import Path
import numpy as np
from neuro_mod.execution.stagers import StageSNNSimulation
from neuro_mod.perturbations import VectorialPerturbation


def _build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a perturbed SNN simulation.")
    parser.add_argument(
        "--config",
        default=str(root / "configs/default_snn_params.yaml"),
        help="Path to the YAML config file.",
    )
    return parser


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    args = _build_parser(root).parse_args()
    config = Path(args.config)
    if not config.is_absolute():
        config = root / config
    stager = StageSNNSimulation(config=config)
    vectors = np.array([
        [1, 1, -1],
        [-1, 1, 1],
        [1, -1, 1],
    ])
    perturbator = VectorialPerturbation(
        *vectors,
        length=38,
        involved_clusters=[[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    )
    params = np.array([0.02, 0.1, 0.05])
    t_vec = np.zeros(int(2.5 // 0.0005))
    t_vec[1000:2000] = 1
    params = np.outer(params, t_vec)
    perturbation = perturbator.get_perturbation(*params)
    outs = stager.run(rate_perturbation=perturbation.T)
    pass
