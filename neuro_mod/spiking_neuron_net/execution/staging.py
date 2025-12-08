
from pathlib import Path

import numpy as np
import torch
import yaml

from neuro_mod.spiking_neuron_net import clustering
from neuro_mod.spiking_neuron_net import external
from neuro_mod.spiking_neuron_net.lif_net import LIFNet


class _StageParent:

    def __init__(
            self,
            delta_t: float,
            duration_sec: float,
            mechanism: str,
            random_seed: int,
            n_neurons: int,
            n_excitatory: int
    ):
        self.delta_t = delta_t
        self.duration_sec = duration_sec
        self.mechanism = mechanism
        self.random_seed = random_seed
        self.n_neurons = n_neurons
        self.n_excitatory = n_excitatory
        self.n_inhibitory = self.n_neurons - self.n_excitatory


class StageSimulation(_StageParent):

    def __init__(self, config: Path | str):
        with open(config, "r") as f:
            config = yaml.safe_load(f)
            super().__init__(**config['simulation'])
            self.settings = config['settings']
            self.network_params = config['architecture']['network']
            self.clusters_params = config['architecture']['clusters']
            self.stimulus_params = config['stimulus']
            self.external_currents_params = config['external_currents']
        self.rng = np.random.default_rng(self.random_seed)
        self.weights, self.clusters = self._get_synaptic_weights()
        self.lif_net = self._get_lif_net()
        self.currents_generator = self._get_external_currents_generator()
        self.stimulus_generator = self._get_stimulus_generator()
        self.save_dir = None
        self.data_file = None
        self._set_stage()

    def _get_lif_net(self) -> LIFNet:
        net_params = self.network_params
        for name, param in net_params.items():
            if isinstance(param, float):
                net_params[name] = param
            elif len(param) == self.n_neurons:
                net_params[name] = np.array(param)
            elif len(param) == 2:
                _new = torch.tensor([param[0]] * self.n_excitatory + [param[1]] * self.n_inhibitory, dtype=torch.float64)
                net_params[name] = _new
            elif len(param) == 4:
                pass
            else:
                raise ValueError(f"Invalid {name} provided. Has to be either length 1 or 2")
        params = {
            "synaptic_weights": torch.from_numpy(self.weights),
            "delta_t": self.delta_t,
            **net_params
        }
        return LIFNet(**params)

    def _get_synaptic_weights(self):
        params = {
            "n_neurons": self.n_neurons,
            "n_excitatory": self.n_excitatory,
            **self.clusters_params
        }
        c, j, b, t = clustering.setup_matrices(**params)
        weights, clusters = clustering.generate_clustered_weight_matrix(
            n_neurons=self.n_neurons,
            boundaries=b,
            synaptic_strengths=j,
            connectivity=c,
            random_generator=self.rng,
            n_excitatory_background=self.clusters_params['n_excitatory_background'],
            n_inhibitory_background=self.clusters_params['n_inhibitory_background'],
            types=t
        )
        return weights, clusters

    def _get_external_currents_generator(self):
        params = {
            "n_neurons": self.n_neurons,
            "n_excitatory": self.n_excitatory,
            "random_generator": self.rng,
            **self.external_currents_params
        }
        return external.ExternalCurrentsGenerator(**params)

    @staticmethod
    def _get_stimulus_generator():
        return external.StimulusGenerator()

    def _set_stage(self):
        self.save_dir = Path(self.settings["save_dir"]) / self.settings["sim_name"]
        self.save_dir.mkdir(parents=True, exist_ok=True)

        plot_dir = self.save_dir / "plots"
        plot_dir.mkdir(exist_ok=True)

        data_dir = self.save_dir / "data"
        data_dir.mkdir(exist_ok=True)

    def _gen_stimulus(self):
        params = {
            "total_duration": self.duration_sec,
            "delta_t": self.delta_t,
            "n_neurons": self.n_neurons,
            **self.stimulus_params
        }
        return self.stimulus_generator.generate_stimulus(**params)

    def execute(self):

        # ----- 1. Generate stimulus and external currents:

        stimulus = torch.from_numpy(self._gen_stimulus())
        external_currents = self.currents_generator.generate_external_currents(duration=self.duration_sec,
                                                                               delta_t=self.delta_t,)
        external_currents = torch.from_numpy(external_currents)

        # ----- 2. Initial conditions:

        voltage = torch.zeros(self.n_neurons, dtype=torch.float64)
        current = torch.zeros(self.n_neurons, dtype=torch.float64)

        # ----- 3. Run:
        voltage, current, spikes = self.lif_net(voltage=voltage,
                                                synaptic_current=current,
                                                stimulus=stimulus,
                                                external_currents=external_currents,
                                                mechanism=self.mechanism)
        voltage = voltage.detach().cpu().numpy()
        voltage = np.astype(voltage, np.float64)
        current = current.detach().cpu().numpy()
        current = np.astype(current, np.float64)
        spikes = spikes.detach().cpu().numpy()
        spikes = np.astype(spikes, np.bool)
        # ----- 4. Save data:
        self._save(voltage=voltage, current=current, spikes=spikes)
        # ----- 5. Plot:
        self._plot(spikes=spikes)

    def _save(self, voltage, current, spikes):
        self.data_file = self.save_dir / "data" / "outputs.npz"
        np.savez(self.data_file,
                 voltage=voltage,
                 current=current,
                 spikes=spikes,
                 clusters=self.clusters,
                 weights=self.weights)

    def _plot(self, spikes: np.ndarray):
        import matplotlib.pyplot as plt
        times, neurons = spikes.nonzero()
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.scatter(times * self.delta_t, neurons, s=0.1, color='black')  # each spike = one dot

        population_markers = [
            (0, self.n_excitatory, 'skyblue', 'Excitatory'),
            (self.n_excitatory, self.n_neurons, 'salmon', 'Inhibitory'),
            (self.n_excitatory - self.clusters_params["n_excitatory_background"],
             self.n_excitatory, 'blue', 'background excitatory'),
            (self.n_neurons - self.clusters_params["n_inhibitory_background"],
             self.n_neurons, 'red', 'background inhibitory'),
        ]
        for y_min, y_max, color, label in population_markers:
            ax.axhspan(ymin=y_min, ymax=y_max, color=color, alpha=0.3,
                       label=label)

        ax.set_xlabel('Time [S]', fontsize=16)
        ax.set_ylabel('Neuron index', fontsize=16)
        ax.set_title('Spike Raster Plot', fontsize=24)
        ax.set_xlim(0, self.duration_sec)
        ax.set_ylim(0, self.n_neurons)
        # fig.gca().invert_yaxis()
        fig.tight_layout()

        plt_path = self.save_dir / "plots" / "spike_raster.png"
        fig.savefig(plt_path)
