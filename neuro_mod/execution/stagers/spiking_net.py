"""Stager for spiking network simulations."""

from pathlib import Path
from numbers import Real
import numpy as np
import torch

from neuro_mod.execution.stagers._base import _Stager
import neuro_mod.core.clustering as clustering
from neuro_mod.core.spiking_net import external
from neuro_mod.core.spiking_net import LIFNet
from neuro_mod.core.spiking_net.core.external import CurrentGenerator


DEFAULT_PARAMS = '../../../configs/default_snn_params.yaml'


class StageSNNSimulation(_Stager):
    """Run a spiking network simulation from configuration."""

    stimulated_clusters: np.ndarray | list

    def __init__(self,
                 config: Path | str | bytes = DEFAULT_PARAMS,
                 random_seed: int = None,
                 output_keys: list[str] | None = None,
                 compile_net: bool = False,
                 **kwargs):
        """Initialize the spiking network stager.

        Args:
            config: Path, string, or bytes for the YAML config.
            random_seed: Optional random seed.
            **kwargs: Extra parameters forwarded to the logic stager.
        """
        super().__init__(config, random_seed, **kwargs)
        self._output_keys = set(output_keys) if output_keys is not None else None
        self._compile_net = compile_net
        self.stimulus_params = self._reader('stimulus')
        self.external_currents_params = self._reader('external_currents')
        self.weights, self.clusters = self._get_synaptic_weights()
        self.stimulus_generator = self._get_stimulus_generator()
        self.current_generator = self._get_currents_generator()

    def _get_lif_net(self) -> LIFNet:
        net_params = self._normalize_net_params(dict(self.network_params))
        params = {
            "synaptic_weights": torch.from_numpy(self.weights),
            "delta_t": self.delta_t,
            "currents_generator": self.current_generator,
            **net_params
        }
        net = LIFNet(**params)
        return self._maybe_compile_net(net)

    def _normalize_net_params(self, net_params: dict) -> dict:
        normalized = {}
        for name, param in net_params.items():
            if isinstance(param, Real):
                normalized[name] = float(param)
                continue
            try:
                length = len(param)
            except TypeError:
                normalized[name] = param
                continue
            if length == self.n_neurons:
                normalized[name] = np.array(param)
            elif length == 2:
                normalized[name] = torch.tensor(
                    [param[0]] * self.n_excitatory + [param[1]] * self.n_inhibitory,
                    dtype=torch.float64,
                )
            elif length == 4:
                normalized[name] = param
            else:
                raise ValueError(f"Invalid {name} provided. Has to be either length 1 or 2")
        return normalized

    def _expand_cluster_vector(self, vec: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_neurons, dtype=float)
        for i, value in enumerate(vec):
            left = self.cluster_vec[i]
            right = self.cluster_vec[i + 1]
            out[left:right] = value
        return out

    def _get_lif_net_with_perturbations(self, perturbations: dict) -> LIFNet:
        """Build a LIF network with cluster-space perturbations applied.

        Args:
            perturbations: Mapping of parameter names to perturbation arrays.

        Returns:
            Initialized LIFNet with perturbed parameters.
        """
        n_pops = len(self.cluster_vec) - 1
        arr_params = {
            "n_populations": n_pops,
            "n_excitatory": self.clusters_params["n_clusters"] + 1,
        }

        def _to_cluster_vector(value):
            arr = np.asarray(value, dtype=float)
            if arr.ndim == 0:
                return np.repeat(arr, n_pops)
            return self._project_to_cluster_space(value, **arr_params)

        net_params = dict(self.network_params)
        for key in ("j_ext", "threshold", "tau_membrane", "tau_synaptic", "tau_refractory"):
            if key not in perturbations:
                continue
            self.logger.debug(
                f"Applying {key} perturbation: "
                f"{self._summarize_perturbation(perturbations[key])}"
            )
            base = _to_cluster_vector(net_params[key])
            base = base + self._coerce_cluster_vector(perturbations[key], n_pops)
            net_params[key] = self._expand_cluster_vector(base)

        net_params = self._normalize_net_params(net_params)

        params = {
            "synaptic_weights": torch.from_numpy(self.weights),
            "delta_t": self.delta_t,
            "currents_generator": self.current_generator,
            **net_params
        }
        net = LIFNet(**params)
        return self._maybe_compile_net(net)

    def _maybe_compile_net(self, net: LIFNet) -> LIFNet:
        if not self._compile_net:
            return net
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            self.logger.warning("torch.compile not available; running uncompiled.")
            return net
        try:
            return compile_fn(net)
        except Exception as exc:
            self.logger.warning("torch.compile failed; running uncompiled. (%s)", exc)
            return net

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
        from neuro_mod.core.spiking_net.analysis.plotting import gen_raster_plot
        spikes = args[0]
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
        plt_path = kwargs.get("plt_path", self.plots_dir / "spike_raster.png")
        fig.savefig(plt_path)

    def run(self, *args, **kwargs):
        """Execute the spiking network simulation.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Optional overrides such as `duration_sec`, `delta_t`,
                and `rate_perturbation`.

        Returns:
            Dictionary of outputs including voltage, current, spikes, and
            stimulus.
        """
        self.logger.info("Running spiking network simulation.")
        perturbations = self._extract_perturbations(kwargs)
        self.duration_sec = kwargs.get("duration_sec", self.duration_sec)
        self.delta_t = kwargs.get("delta_t", self.delta_t)
        stimulus = torch.from_numpy(self._gen_stimulus())
        lif_net = self._get_lif_net_with_perturbations(perturbations) if perturbations else self._get_lif_net()
        perturbation_shape = (stimulus.shape[0], len(np.unique(self.clusters)))
        rate_source = None
        rate_perturbation = perturbations.get('rate')
        if rate_perturbation is not None:
            rate_source = "kwargs"
        if rate_perturbation is None:
            rate_perturbation = self._init_perturbations.get('rate')
            if rate_perturbation is not None:
                rate_source = "config"
        if rate_perturbation is None:
            rate_perturbation = kwargs.get('rate_perturbation')
            if rate_perturbation is not None:
                rate_source = "kwargs_rate_perturbation"
        if rate_perturbation is None:
            rate_source = "default_zeros"
            rate_perturbation = np.zeros(perturbation_shape)
        rate_perturbation = np.asarray(rate_perturbation, dtype=float)
        if rate_perturbation.ndim == 1 and rate_perturbation.shape[0] == perturbation_shape[1]:
            rate_perturbation = np.tile(rate_perturbation, (perturbation_shape[0], 1))
        elif rate_perturbation.ndim == 2 and rate_perturbation.shape[0] == perturbation_shape[1]:
            rate_perturbation = rate_perturbation.T
        self.logger.debug(
            f"Rate perturbation source={rate_source} "
            f"summary={self._summarize_perturbation(rate_perturbation)}"
        )
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
        if self._output_keys is not None:
            outputs = {key: value for key, value in outputs.items() if key in self._output_keys}
        self.logger.info("Spiking network simulation complete.")
        return outputs
