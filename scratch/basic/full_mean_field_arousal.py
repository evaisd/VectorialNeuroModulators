
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from neuro_mod.execution.sweep import FullMeanFieldBaseSweepRunner


class ArousalSweepRunner(FullMeanFieldBaseSweepRunner):

    def _summary_plot(self, *args, **kwargs):
        pass

    def __init__(self):
        super().__init__()

    def _step(self,
              param: str,
              sweep_param: float,
              nu_init: np.ndarray = None,
              *args,
              **kwargs):
        outputs = self._sweep_object.run(*args, nu_init=nu_init)
        e_clusters = self._sweep_object.clusters_params['n_clusters']
        e_cluster_active_rates, e_cluster_non_active_rates = self._analysis(outputs['fixed_point'][:, :e_clusters])
        self._store(sweep_param, **outputs)
        return e_cluster_active_rates, e_cluster_non_active_rates

    def _store(self, arousal_level, **kwargs):
        data_file = self._dirs['data'].joinpath(f"arousal_level_{arousal_level:.2f}.npz")
        np.savez(data_file,
                 **kwargs)

    @staticmethod
    def _analysis(rates: np.ndarray, z_threshold: float = 2.):
        from neuro_mod.mean_field.analysis.activity import classify_active_clusters
        active_clusters = classify_active_clusters(rates, z_threshold)
        return rates[active_clusters].mean(), rates[~active_clusters].mean()

    def summary(self, *args, **kwargs):
        import matplotlib.pyplot as plt
        results, sweep_params = args
        results = np.stack(results, axis=1)
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(sweep_params, results[0], 'o-', label='active')
        ax.plot(sweep_params, results[1], 'o-', label='non-active')
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


def main():
    config = 'configs/default_mf_params.yaml'
    arousal_params = np.linspace(1e-6, 1., 30)
    wd = Path(next(p for p in Path().resolve().parents if p.name == 'VectorialNeuroModulators'))
    os.chdir(wd)
    runner = ArousalSweepRunner()
    runner.set_dirs('simulations/sweep_arousal_mf')
    runner.execute('simulations/sweep_arousal_mf',
                   config,
                   param=['arousal', 'level'],
                   sweep_params=arousal_params,)


if __name__ == '__main__':
    main()
