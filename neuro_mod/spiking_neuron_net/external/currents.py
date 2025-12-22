
from functools import lru_cache
import numpy as np


def generate_external_currents(
        nu_ext_baseline: list[float] | list[float] | np.ndarray,
        c_ext: np.ndarray[int] | list[int] | np.ndarray,
        duration: float,
        delta_t: float,
        cluster_vec: list[int] | np.ndarray[int],
        n_e_clusters: int ,
        cluster_based_perturbations: list[float] | np.ndarray[float] = None,
        neuron_based_perturbations: list[float] | np.ndarray[float] = None,
        rng: np.random.Generator = None,
        *args,
        **kwargs
):
    total_steps = int(duration / delta_t)
    full_output = np.zeros((4, total_steps, cluster_vec[-1]))
    projected_nu = _project_to_cluster_space(len(cluster_vec) - 1,
                                             n_e_clusters,
                                             nu_ext_baseline)
    projected_c = _project_to_cluster_space(len(cluster_vec) - 1,
                                            n_e_clusters,
                                            c_ext)
    poisson = [[], []]
    if cluster_based_perturbations is None:
        cluster_based_perturbations = np.zeros(len(cluster_vec) - 1)
    if neuron_based_perturbations is None:
        neuron_based_perturbations = np.zeros(cluster_vec[-1])


    for i in range(2):
        iterator = zip(cluster_vec[:-1], cluster_vec[1:],
                       cluster_based_perturbations)
        for j, (left, right, p) in enumerate(iterator):
            num_neurons_to = right - left
            base_rate = projected_nu[i, j]
            c = projected_c[i, j]
            delta_nu = neuron_based_perturbations[left:right]
            rate = (base_rate + delta_nu + p) * c * delta_t
            size = (total_steps, num_neurons_to)
            poisson[i].append(_gen_poisson(rate, size, rng))
    full_output[0] = np.concatenate(poisson[0], axis=1)
    full_output[1] = np.concatenate(poisson[1], axis=1)
    return full_output.swapaxes(0, 1)


def _gen_poisson(rate, size, rng=None):
    if rng is None:
        rng = np.random.default_rng(256)
    rate = np.maximum(rate, 0.)
    return rng.poisson(rate, size)


def _project_to_cluster_space(
        n_clusters: int,
        n_e_clusters: int,
        param: list[float | int] | list[float | int],
):
    if len(param) == 1:
        param = param * 2 + [0, 0]
    projected = np.zeros((2, n_clusters))

    projected[0, :n_e_clusters] = param[0]
    projected[1, :n_e_clusters] = param[2]
    projected[0, n_e_clusters:] = param[1]
    projected[1, n_e_clusters:] = param[3]
    return projected


class CurrentGenerator:

    def __init__(
            self,
            rng: np.random.Generator,
            delta_t: float,
            cluster_vec: list[int] | np.ndarray[int],
            n_e_clusters: int,
            n_neurons: int,
            c_ext: np.ndarray[int] | list[int],
    ):
        self.rng = rng
        self.delta_t = delta_t
        self.sizes = np.diff(cluster_vec)
        self.n_clusters = len(self.sizes)
        self.cluster_vec = cluster_vec
        self.n_e_clusters = n_e_clusters
        self.n_neurons = n_neurons
        if len(c_ext) == 4:
            self.c_ext = self._project_to_cluster_space(c_ext)
        else:
            raise ValueError("c_ext must be length 4")

    def _project_to_cluster_space(
            self,
            param: list[float] | list[float] | np.ndarray,
    ):
        return _project_to_cluster_space(self.n_clusters, self.n_e_clusters, param)

    def _gen(self, rate, size):
        return _gen_poisson(rate, size, rng=self.rng)

    def generate_currents(
            self,
            baseline_rates: list[float] | np.ndarray[float],  # (4,)
            c_perturbations: list[float] | np.ndarray[float] = None,  # (C, T)
            n_perturbations: list[float] | np.ndarray[float] = None   # (T,)
    ):
        baseline_rates = self._project_to_cluster_space(baseline_rates)
        c_perturbations = np.zeros((self.n_clusters, 1)) if c_perturbations is None else c_perturbations
        n_perturbations = np.zeros(self.n_neurons) if n_perturbations is None else n_perturbations
        full_output = np.zeros((4, self.n_neurons, ))
        poisson = [[], []]
        for i in range(2): # EE, EI
            for j, size in enumerate(self.sizes):  # clusters
                baseline_rate = baseline_rates[i, j]
                perturbation = (c_perturbations[j] +
                                n_perturbations[self.cluster_vec[j]:self.cluster_vec[j + 1]])
                rate = (baseline_rate + perturbation) * self.delta_t * self.c_ext[i, j]
                poisson[i].append(_gen_poisson(rate.flatten(), size, self.rng))
        full_output[0] = np.concatenate(poisson[0], axis=0)
        full_output[1] = np.concatenate(poisson[1], axis=0)
        return full_output
