
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from neuro_mod.execution.sweep import SNNBaseSweepRunner
from neuro_mod.spiking_neuron_net.analysis.plotting import gen_raster_plot


class PerturbedSweepSNN(SNNBaseSweepRunner):

    def _summary_plot(self, *args, **kwargs):
        pass

    def summarize_repeated_run(self, *args, **kwargs):
        pass

    def summary(self, *args, **kwargs):
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _step(self, *args, **kwargs):
        outputs = self._sweep_object.run()
        self._store(*args, **outputs)
        return

    def _store(self, *args, **kwargs):
        param, idx, sweep_param = args
        param_name = '_'.join(param)
        data_file = self._dirs['data'].joinpath(f"{param_name}_{sweep_param:.2f}.npz")

        np.savez(data_file,
                 kwargs['spikes'].astype(bool),
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

        fig_path = self._dirs['plots'].joinpath(f"{param_name}_{sweep_param:.2f}_raster.png")
        fig.savefig(fig_path)
        plt.close(fig)


def _build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep SNN perturbation strengths.")
    parser.add_argument(
        "--config",
        default=str(root / "configs/perturbed/default_snn_params_with_perturbation.yaml"),
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--save-dir",
        default=str(root / "simulations/sweep_arousal_snn_perturbed/test"),
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
    rate_params = np.linspace(1e-6, 2, args.n_steps)
    runner = PerturbedSweepSNN()
    runner.execute(save_dir,
                   config,
                   param=['perturbation', 'params'],
                   param_idx=0,
                   sweep_params=rate_params)




if __name__ == '__main__':
    main()
