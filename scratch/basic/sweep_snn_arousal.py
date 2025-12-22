
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from neuro_mod.execution.sweep import SNNBaseSweepRunner
from neuro_mod.spiking_neuron_net.analysis.activity import get_cluster_activity
from neuro_mod.spiking_neuron_net.analysis.plotting import gen_raster_plot


class ArousalSweepRunner(SNNBaseSweepRunner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def _step(self, *args, **kwargs):
        outputs = self._sweep_object.run()
        e_cluster_active_rates, e_cluster_non_active_rates = self._analysis(outputs['spikes'],
                                                                            outputs['clusters'])
        self._store(*args, **outputs)
        return e_cluster_active_rates, e_cluster_non_active_rates

    def _store(self, *args, **kwargs):
        param, idx, arousal_level = args
        param_name = '_'.join(param)
        data_file = self._dirs['data'].joinpath(f"{param_name}_{arousal_level:.2f}.npz")

        np.savez(data_file,
                 kwargs['spikes'].astype(np.bool),
                 kwargs['clusters'].astype(np.uint8), )
        pops = (
            self._sweep_object.n_excitatory,
            self._sweep_object.n_excitatory - self._sweep_object.clusters_params["n_excitatory_background"],
            self._sweep_object.n_neurons - self._sweep_object.clusters_params["n_inhibitory_background"]
        )
        delta_t = self._sweep_object.delta_t
        duration_sec = self._sweep_object.duration_sec
        fig = gen_raster_plot(
            kwargs['spikes'],
            delta_t,
            duration_sec,
            self._sweep_object.n_neurons,
            *pops
        )

        fig_path = self._dirs['plots'].joinpath(f"{param_name}_{arousal_level:.2f}_raster.png")
        fig.savefig(fig_path)
        plt.close(fig)

    def _analysis(self,
                  spikes: np.ndarray[bool],
                  clusters: np.ndarray[np.uint8],
                  *args,
                  **kwargs):
        n_clusters = self._sweep_object.clusters_params['n_clusters']
        window_ms = kwargs.get("window_ms", 10.)
        frate, active = get_cluster_activity(spikes.T,
                                             clusters,
                                             dt_ms=self._sweep_object.delta_t / 1e-3,
                                             kernel_param=window_ms,
                                             kernel_type='gaussian')
        if active[:n_clusters].sum() > 0:
            e_cluster_active_rates = frate[:n_clusters][active[:n_clusters]].mean()
        else:
            e_cluster_active_rates = 0
        e_cluster_non_active_rates = frate[:n_clusters][~active[:n_clusters]].mean()
        return e_cluster_active_rates, e_cluster_non_active_rates

    def summary(self, *args, **kwargs):
        results, sweep_params = args
        results = np.stack(results, axis=1)
        self._summary_plot(results, sweep_params)
        np.savez(self._dirs['data'].joinpath(f"summary_results.npz"),
                 summary_results=results)

    def _summary_plot(self, *args, **kwargs):
        fig, ax = plt.subplots(figsize=(16, 8))
        results, sweep_params = args
        ax.plot(sweep_params, results[0], 'o--', label="Active")
        ax.plot(sweep_params, results[1], 'o--', label="Non-Active")
        ax.legend()
        ax.set_xlabel("Arousal Level")
        ax.set_ylabel("E cluster Spike Rate / S")
        fig.savefig(self._dirs['plots'].joinpath(f"active_vs_non_active.png"))
        plt.close(fig)

    def summarize_repeated_run(self, *args, **kwargs):
        pass


def main():
    config = 'configs/default_snn_params.yaml'
    arousal_params = np.linspace(1e-6, 1., 30)
    wd = Path(next(p for p in Path().resolve().parents if p.name == 'VectorialNeuroModulators'))
    os.chdir(wd)
    runner = ArousalSweepRunner()
    runner.execute('simulations/sweep_arousal_snn/test',
                   config, param=['arousal', 'level'],
                   sweep_params=arousal_params)


if __name__ == '__main__':
    main()