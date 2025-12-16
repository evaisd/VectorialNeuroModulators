
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sweep_snn_arousal import ArousalSweepRunner
from neuro_mod.spiking_neuron_net.analysis.activity import get_average_cluster_firing_rate


class ArousalRepeatedSweepRunner(ArousalSweepRunner):

    def _step(self, *args, **kwargs):
        outputs = self._sweep_object.run()
        firing_rates = self._analysis(outputs['spikes'], outputs['clusters'])
        self._store(*args, **outputs)
        return firing_rates

    def _analysis(self,
                  spikes: np.ndarray[bool],
                  clusters: np.ndarray[np.uint8],
                  *args,
                  **kwargs):
        window_ms = kwargs.get("window_ms", 10.)
        kernel_type = kwargs.get("kernel_type", "gaussian")
        firing_rates = get_average_cluster_firing_rate(
            spikes.T,
            clusters,
            dt_ms=self._sweep_object.delta_t / 1e-3,
            kernel_param=window_ms,
            kernel_type=kernel_type
        )
        return firing_rates

    def _summary_plot(self, *args, **kwargs):
        pass

    def _summarize_repeated_run(self, *args, **kwargs):
        directory, sweep_params = args
        trials = self._dirs['main'].glob('*')
        all_spikes = []
        for trial in trials:
            file = trial.joinpath('data', 'summary_results.npz')
            all_spikes.append(np.load(file)['summary_results'])
        all_spikes = np.concat(all_spikes, axis=-1)[:18]
        average_cluster_rates = np.mean(all_spikes, axis=-1)
        active = all_spikes > average_cluster_rates[..., None]
        masked_sum = np.sum(np.where(active, all_spikes, 0.0), axis=(1, 2))
        masked_count = np.sum(active, axis=(1, 2))
        means_active = masked_sum / masked_count
        masked_sum = np.sum(np.where(~active, all_spikes, 0.0), axis=(1, 2))
        masked_count = np.sum(active, axis=(1, 2))
        means_non_active = masked_sum / masked_count
        file = directory.joinpath('summary_results.npz')
        np.savez(file,
                 means_active=means_active,
                 means_non_active=means_non_active)
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(sweep_params, means_active, '--o', label='Active')
        ax.plot(sweep_params, means_non_active, '--o', label='Non-Active')
        ax.legend()
        ax.set_xlabel("Arousal Level")
        ax.set_ylabel("E cluster Spike Rate / S")
        fig.savefig(directory.joinpath(f"active_vs_non_active.png"))
        plt.close(fig)

def main():
    config = 'configs/18_clusters_snn.yaml'
    arousal_params = np.linspace(1e-6, 1., 30)
    wd = Path().cwd().parent
    os.chdir(wd)
    runner = ArousalRepeatedSweepRunner()
    runner.repeat('simulations/sweep_arousal_snn/repeat_test',
                  n_trials=30,
                  baseline_params=config,
                   param=['arousal', 'level'],
                   sweep_params=arousal_params)


if __name__ == '__main__':
    main()