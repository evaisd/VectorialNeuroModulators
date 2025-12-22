
import yaml
from pathlib import Path
import numpy as np
import torch

from neuro_mod.execution.stagers._base import _Stager
from neuro_mod import clustering
from neuro_mod.spiking_neuron_net import external
from neuro_mod.spiking_neuron_net.lif_net import LIFNet
from neuro_mod.spiking_neuron_net.external import CurrentGenerator


DEFAULT_PARAMS = '../../../configs/default_snn_params.yaml'


class StageSNNSimulation(_Stager):

    def __init__(self,
                 config: Path | str | bytes = DEFAULT_PARAMS,
                 random_seed: int = None,
                 **kwargs):
        super().__init__(config, random_seed, **kwargs)
        self.stimulus_params = self._reader('stimulus')
        self.external_currents_params = self._reader('external_currents')
        self.weights, self.clusters = self._get_synaptic_weights()
        self.stimulus_generator = self._get_stimulus_generator()
        self.stimulated_clusters = None
        self.current_generator = self._get_currents_generator()

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
            "currents_generator": self.current_generator,
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

    def _get_currents_generator(self):

        params = {
            "rng": self.rng,
            "delta_t": self.delta_t,
            "cluster_vec": self.cluster_vec,
            "n_e_clusters":  self.n_excitatory,
            "n_neurons": self.n_neurons,
            "c_ext": self.external_currents_params["c_ext"],
        }
        return CurrentGenerator(**params)

    def _generate_currents(self, perturbation: list[float] | np.ndarray[float] = None):
        delta_nu_ext = self._get_arousal_nu()
        baseline_rates = np.array(self.external_currents_params["nu_ext_baseline"], dtype=np.float64)
        return torch.from_numpy(
            self.current_generator.generate_currents(
                baseline_rates=baseline_rates,
                n_perturbations=delta_nu_ext,
                c_perturbations=perturbation
            )
        )

    @staticmethod
    def _get_stimulus_generator():
        return external.StimulusGenerator()

    def _gen_stimulus(self):
        n_clusters = self.clusters_params['n_clusters']
        stimulus_params = dict(self.stimulus_params)
        stimulated_clusters_prob = stimulus_params.pop("stimulated_clusters_prob")
        self.stimulated_clusters = self.rng.choice(range(1, n_clusters + 1),
                                                   size=int(n_clusters * stimulated_clusters_prob),
                                                   replace=False)
        stimulated_neuron_prob = stimulus_params.pop("stimulated_neuron_prob")
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
            **stimulus_params
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
        self.duration_sec = kwargs.get("duration_sec", self.duration_sec)
        self.delta_t = kwargs.get("delta_t", self.delta_t)
        stimulus = torch.from_numpy(self._gen_stimulus())
        lif_net = self._get_lif_net()
        perturbation_shape = (stimulus.shape[0], len(np.unique(self.clusters)))
        rate_perturbation = kwargs.get('rate_perturbation', np.zeros(perturbation_shape))
        external_currents = (self._generate_currents(c) for c in rate_perturbation)
        voltage = torch.zeros(self.n_neurons, dtype=torch.float64)
        current = torch.zeros(self.n_neurons, dtype=torch.float64)

        voltage, current, spikes = lif_net(voltage=voltage,
                                                synaptic_current=current,
                                                stimulus=stimulus,
                                                external_currents=external_currents,
                                                mechanism=self.mechanism)
        voltage = voltage.detach().cpu().numpy().astype(np.float64)
        current = current.detach().cpu().numpy().astype(np.float64)
        spikes = spikes.detach().cpu().numpy().astype(bool)

        outputs = {
            "voltage": voltage,
            "current": current,
            "spikes": spikes,
            "stimulus": stimulus,
            "weights": self.weights,
            "clusters": self.clusters,
        }
        return outputs
