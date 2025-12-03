
from neuro_mod.mean_field.runners._base import *
from neuro_mod.mean_field.core import LIFMeanField


class MainMFRunner(SimRunner):
    """High-level runner for mean-field fixed-point and stability analysis.

    This class provides a compact API around `LIFMeanField` to:

    * solve for fixed-point firing rates, and
    * classify their stability.

    Args:
        lif_mf: Configured mean-field model instance.
    """

    def __init__(self, lif_mf: LIFMeanField):
        super().__init__(lif_mf)

    def run(self, nu_init, *args, **kwargs):
        """Run the fixed-point solver and stability analysis.

        Args:
            nu_init: Initial guess for firing rates.
            *args: Additional positional arguments (ignored).
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple `(nu_star, fp_type, eigvals)` where:

            * `nu_star` are the fixed-point firing rates.
            * `fp_type` is a string describing the fixed-point type.
            * `eigvals` are the eigenvalues of the Jacobian at the fixed point.
        """
        res = self.lif.solve_rates(nu_init)
        stability = self.lif.determine_stability(res.x)
        return res.x, *stability

    def _gen_params(self, **params):
        """Placeholder for generic parameter handling."""
        pass

    def _mft_params(self, **params):
        """Placeholder for mean-field specific parameter handling."""
        pass

    def _settings(self):
        """Placeholder for run-time settings."""
        pass

    def run_effective_rates_on_grid(self, focus_pops: list[int], *focus_nu_vecs: np.ndarray):
        """Evaluate the effective response on a grid of input rates.

        Args:
            focus_pops: Indices of populations of interest.
            *focus_nu_vecs: 1D vectors of rates defining the grid for each
                focus population.

        Returns:
            Tuple `(mesh, nu_outs)` where:

            * `mesh` is the grid of input rates.
            * `nu_outs` are the corresponding effective output rates.
        """
        mesh = np.concatenate(np.meshgrid(*focus_nu_vecs), axis=-1)
        nu_outs = np.empty_like(mesh)
        for idx in np.ndindex(mesh.shape[:self.lif.n_populations - 1]):
            nu_outs[idx] = self.lif.effective_response_function(focus_pops, mesh[idx])
        return mesh, nu_outs
