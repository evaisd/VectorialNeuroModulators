
import yaml
from pathlib import Path
import numpy as np
import torch

from neuro_mod.execution.stagers._base import _Stager
from neuro_mod import clustering
from neuro_mod.spiking_neuron_net import external
from neuro_mod.spiking_neuron_net.lif_net import LIFNet


class StageSNNSimulation(_Stager):

    def __init__(self,
                 config: Path | str | bytes,
                 random_seed: int = None,
                 **kwargs):
        super().__init__(config, random_seed, **kwargs)
        self.stimulus_params = self._reader('stimulus')
        self.external_currents_params = self._reader('external_currents')
        self.weights, self.clusters = self._get_synaptic_weights()
        self.lif_net = self._get_lif_net()
        self.currents_generator = self._get_external_currents_generator()
        self.stimulus_generator = self._get_stimulus_generator()
        self.stimulated_clusters = None

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
        weights, clusters = clustering.generate_clustered_weight_matrix(
            n_neurons=self.n_neurons,
            boundaries=self.cluster_vec,
            synaptic_strengths=self.j_mat,
            connectivity=self.p_mat,
            random_generator=self.rng,
            n_excitatory_background=self.clusters_params['n_excitatory_background'],
            n_inhibitory_background=self.clusters_params['n_inhibitory_background'],
            types=self.types,
        )
        return weights, clusters

    def _get_external_currents_generator(self):
        delta_nu_ext = self._get_arousal_nu()
        params = {
            "n_neurons": self.n_neurons,
            "n_excitatory": self.n_excitatory,
            "random_generator": self.rng,
            "delta_nu_ext": delta_nu_ext,
            **self.external_currents_params
        }
        return external.ExternalCurrentsGenerator(**params)

    @staticmethod
    def _get_stimulus_generator():
        return external.StimulusGenerator()

    def _gen_stimulus(self):
        n_clusters = self.clusters_params['n_clusters']
        stimulated_clusters_prob = self.stimulus_params.pop("stimulated_clusters_prob")
        self.stimulated_clusters = self.rng.choice(range(1, n_clusters + 1),
                                                   size=int(n_clusters * stimulated_clusters_prob),
                                                   replace=False)
        stimulated_neuron_prob = self.stimulus_params.pop("stimulated_neuron_prob")
        possible_neurons = np.argwhere(np.isin(self.clusters,
                                               self.stimulated_clusters)).flatten()
        stimulated_neurons = self.rng.choice(possible_neurons,
                                             int(len(possible_neurons) * stimulated_neuron_prob),
                                             replace=False)
        params = {
            "n_neurons": self.n_neurons,
            "total_duration": self.duration_sec,
            "delta_t": self.delta_t,
            "stimulated_neurons": stimulated_neurons,
            "duration": self.duration_sec,
            **self.stimulus_params
        }

        return self.stimulus_generator.generate_stimulus(**params)

    def _plot(self, *args, **kwargs):
        from neuro_mod.spiking_neuron_net.analysis.plotting import gen_raster_plot
        spikes = args
        pops = (
            self.n_excitatory,
            self.n_excitatory - self.clusters_params["n_excitatory_background"],
            self.n_neurons - self.clusters_params["n_inhibitory_background"]
        )
        fig = gen_raster_plot(
            spikes,
            self.delta_t,
            self.duration_sec,
            self.n_neurons,
            *pops
        )
        plt_path = self.plots_dir / "spike_raster.png"
        fig.savefig(plt_path)

    def run(self, *args, **kwargs):
        stimulus = torch.from_numpy(self._gen_stimulus())
        external_currents = self.currents_generator.generate_external_currents(duration=self.duration_sec,
                                                                               delta_t=self.delta_t,)
        external_currents = torch.from_numpy(external_currents)

        voltage = torch.zeros(self.n_neurons, dtype=torch.float64)
        current = torch.zeros(self.n_neurons, dtype=torch.float64)

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

        outputs = {
            "voltage": voltage,
            "current": current,
            "spikes": spikes,
            "stimulus": stimulus,
            "weights": self.weights,
            "clusters": self.clusters,
        }
        return outputs
