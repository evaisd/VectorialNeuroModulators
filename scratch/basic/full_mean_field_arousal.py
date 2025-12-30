
import argparse
from pathlib import Path
import numpy as np

from neuro_mod.execution.sweep import FullMeanFieldBaseSweepRunner


class ArousalSweepRunner(FullMeanFieldBaseSweepRunner):

    def _summary_plot(self, *args, **kwargs):
        pass

    def __init__(self):
        super().__init__()

    def _step(self,
              param: str,
              idx: int,
              sweep_param: float,
              *args,
              **kwargs):
        outputs = self._sweep_object.run(*args)
        e_clusters = self._sweep_object.clusters_params['n_clusters']
        active_rates, non_active_rates, uniform_rates = self._analysis(outputs['fixed_point'][:, :e_clusters])
        self._store(sweep_param, **outputs)
        return active_rates, non_active_rates, uniform_rates

    def _store(self, arousal_level, **kwargs):
        data_file = self._dirs['data'].joinpath(f"arousal_level_{arousal_level:.2f}.npz")
        np.savez(data_file,
                 **kwargs)

    @staticmethod
    def _analysis(rates: np.ndarray, z_threshold: float = 2.):
        from neuro_mod.mean_field.analysis.logic.activity import classify_active_clusters
        active_clusters = classify_active_clusters(rates, z_threshold)
        return active_clusters

    def summary(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        results, sweep_params = args
        results = np.stack(results, axis=1)
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(sweep_params, results[0], 'o-', label='active')
        ax.plot(sweep_params, results[1], 'o-', label='non-active')
        ax.plot(sweep_params, results[2], 'o-', label='uniform')
        ax.legend()
        ax.set_xlabel("Arousal Level")
        ax.set_ylabel("E cluster Spike Rate / S")
        fig.savefig(self._dirs['plots'].joinpath(f"active_vs_non_active.png"))
        plt.close(fig)
        np.savez(self._dirs['data'].joinpath("active_vs_non_active.npz"),
                 active_rates=results[0],
                 non_active_rates=results[1],)

    def summarize_repeated_run(self, *args, **kwargs):
        pass


def _build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep mean-field arousal levels.")
    parser.add_argument(
        "--config",
        default=str(root / "configs/2_cluster_mf.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(root / "simulations/sweep_arousal_mf"),
        help="Output directory for sweep artifacts.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=30,
        help="Number of sweep values.",
    )
    return parser


def main():
    root = Path(__file__).resolve().parents[2]
    args = _build_parser(root).parse_args()
    config = Path(args.config)
    if not config.is_absolute():
        config = root / config
    save_dir = Path(args.save_dir)
    if not save_dir.is_absolute():
        save_dir = root / save_dir
    arousal_params = np.linspace(1e-6, 1., args.n_steps)
    runner = ArousalSweepRunner()
    runner.set_dirs(save_dir)
    runner.execute(save_dir, config, param=['arousal', 'level'], sweep_params=arousal_params)


if __name__ == '__main__':
    main()
