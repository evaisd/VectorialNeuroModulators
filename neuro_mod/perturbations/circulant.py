from scipy.linalg import circulant
import numpy as np

from neuro_mod.perturbations._base import BasePerturbator


class CirculantPerturbation(BasePerturbator):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(**kwargs)
        self.n_clusters = kwargs.get('n_clusters')
        self.n_excitatory = kwargs.get('n_excitatory')
        self.n_clusters_to_perturb = kwargs.get('n_clusters_to_perturb', self.n_clusters)

    def get_perturbation(self, *params, **kwargs):
        alpha, beta = params
        perturbation = np.ones((self.n_clusters, self.n_clusters))
        base_pert = circulant(np.eye(self.n_clusters_to_perturb)[-1])
        pert_exc = alpha * base_pert
        pert_inh = beta * base_pert.T
        slice_exc = slice(0, self.n_clusters_to_perturb)
        slice_inh = slice(self.n_excitatory,
                          self.n_excitatory + self.n_clusters_to_perturb)
        perturbation[slice_exc, slice_exc] += pert_exc
        perturbation[slice_exc, slice_inh] += pert_inh
        return perturbation
