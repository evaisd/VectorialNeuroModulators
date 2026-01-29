
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from neuro_mod.execution.sweep import SNNBaseSweepRunner
from neuro_mod.spiking_neuron_net.analysis.logic import activity
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
        param, idx, sweep_param = args
        param_name = '_'.join(param)
        data_file = self._dirs['data'].joinpath(f"{param_name}_{sweep_param:.2f}.npz")

        np.savez(data_file,
                 spikes=kwargs['spikes'].astype(bool),
                 clusters=kwargs['clusters'].astype(np.uint8), )
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

        fig_path = self._dirs['plots'].joinpath(f"{param_name}_{sweep_param:.2f}_raster.png")
        fig.savefig(fig_path)
        plt.close(fig)

    def _analysis(self,
                  spikes: np.ndarray[bool],
                  clusters: np.ndarray[np.uint8],
                  *args,
                  **kwargs):
        n_clusters = self._sweep_object.clusters_params['n_clusters']
        window_ms = kwargs.get("window_ms", 10.)
        frate = activity.get_average_cluster_firing_rate(
            spikes,
            clusters,
            dt_ms=self._sweep_object.delta_t / 1e-3,
            kernel_param=window_ms,
            kernel_type=kwargs.get('kernel_type', 'gaussian'),
        )
        active = activity.get_activity(
            frate,
            kwargs.get("baseline_rate", None)
        )
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


def _build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep SNN arousal levels.")
    parser.add_argument(
        "--config",
        default=str(root / "configs/default_snn_params.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(root / "simulations/sweep_arousal_snn/test"),
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
    runner.execute(save_dir, config, param=['arousal', 'level'],
                   sweep_params=arousal_params)


if __name__ == '__main__':
    main()
