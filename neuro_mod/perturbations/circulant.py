"""Circulant perturbation generator for clustered connectivity."""

from scipy.linalg import circulant
import numpy as np

from neuro_mod.perturbations._base import BasePerturbator


class CirculantPerturbation(BasePerturbator):
    """Generate a circulant block perturbation for clustered networks."""

    def __init__(self,
                 *args,
                 **kwargs):
        """Initialize the circulant perturbation parameters.

        Args:
            *args: Ignored positional arguments for compatibility.
            **kwargs: Keyword arguments with required sizes:
                n_clusters: Total number of clusters.
                n_excitatory: Number of excitatory clusters.
                n_clusters_to_perturb: Number of clusters to perturb.
                rng: Optional NumPy random generator.
        """
        super().__init__(**kwargs)
        self.n_clusters = kwargs.get('n_clusters')
        self.n_excitatory = kwargs.get('n_excitatory')
        self.n_clusters_to_perturb = kwargs.get('n_clusters_to_perturb', self.n_clusters)

    def get_perturbation(self, *params, **kwargs):
        """Build the circulant perturbation matrix.

        Args:
            *params: Two coefficients `(alpha, beta)` for excitatory and
                inhibitory perturbations.
            **kwargs: Ignored keyword arguments for compatibility.

        Returns:
            A `(n_clusters, n_clusters)` perturbation matrix.
        """
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
