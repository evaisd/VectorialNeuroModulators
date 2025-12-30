"""Stagers for mean-field simulations."""

from abc import ABC
from pathlib import Path
import numpy as np
from neuro_mod.execution.stagers._base import _Stager
from neuro_mod.mean_field import LIFMeanField


class _BaseMeanFieldStager(_Stager, ABC):

    def __init__(self,
                 config_path: Path | str | bytes,
                 random_seed: int = None,
                 **kwargs):
        super().__init__(config_path, random_seed, **kwargs)
        self.external_currents_params = self._reader('external_currents')
        self.c_mat = np.zeros_like(self.p_mat)
        self._set_c_matrix()
        self.lif_mf = self._get_mf_lif(kwargs.get("rate_perturbation", 0.))

    def _get_mf_lif(self, rate_perturbation=0.):
        j_ext = self.network_params['j_ext']
        j_ext = np.asarray(j_ext) / self.n_neurons ** .5
        nu_ext = self.external_currents_params['nu_ext_baseline']
        delta_nu_ext_full = self._get_arousal_nu()
        delta_nu_ext = np.zeros_like(nu_ext)
        delta_nu_ext[0] = delta_nu_ext_full[:self.n_excitatory].mean()
        delta_nu_ext[1] = delta_nu_ext_full[self.n_excitatory:].mean()
        nu_ext += delta_nu_ext
        arr_params = {
            "n_populations": self.j_mat.shape[0],
            "n_excitatory": self.clusters_params['n_clusters'] + 1
        }
        j_ext, c_ext, nu_ext = [self._project_to_cluster_space(x, **arr_params)
                                for x in
                                (j_ext, self.external_currents_params['c_ext'], nu_ext)]
        nu_ext += rate_perturbation
        params = {
            "n_clusters": self.clusters_params['n_clusters'],
            'c_matrix': self.c_mat,
            'j_matrix': self.j_mat,
            'j_ext': j_ext,
            'c_ext': c_ext,
            'nu_ext': nu_ext,
            'tau_membrane': self.network_params['tau_membrane'],
            'tau_synaptic': self.network_params['tau_synaptic'],
            'threshold': self.network_params['threshold'],
            'reset_voltage': self.network_params['reset_voltage'],
            'tau_refractory': self.network_params['tau_refractory'],
        }
        return LIFMeanField(**params)

    def _set_c_matrix(self):
        pops = np.diff(self.cluster_vec)
        iterator = np.ndenumerate(self.p_mat)
        for (i, j), _ in iterator:
            self.c_mat[i, j] = self.p_mat[i, j] * pops[j]


class FullMeanFieldStager(_BaseMeanFieldStager):
    """Run full mean-field fixed-point simulations."""

    def _plot(self, *args, **kwargs):
        pass

    def __init__(self,
                 config: str | Path | bytes,
                 random_seed: int = None,
                 **kwargs):
        """Initialize the full mean-field stager.

        Args:
            config: Path, string, or bytes for the YAML config.
            random_seed: Optional random seed.
            **kwargs: Extra parameters forwarded to the base stager.
        """
        super().__init__(config, random_seed, **kwargs)

    def run(self,
            nu_init: np.ndarray | None = None,
            n_runs: int = 100,
            *args,
            **kwargs):
        """Solve mean-field fixed points from multiple initial conditions.

        Args:
            nu_init: Optional initial rates; if None, random initializations.
            n_runs: Number of random initializations.
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.

        Returns:
            Dictionary with fixed points and corresponding initializations.
        """
        self.logger.info(f"Running full mean-field simulation with {n_runs} runs.")
        if nu_init is None:
            nu_init = self._draw_nu_inits(n_runs)
        nu_init = np.atleast_2d(nu_init)
        ress, n_i = [], []
        for run, nu in enumerate(nu_init):
            res = self.lif_mf.solve_rates(nu).x
            stability = self.lif_mf.determine_stability(res)[0]
            ress.append(res)
            n_i.append(nu)
        output = {
            "fixed_point": np.stack(ress, axis=0),
            "nu_init": np.stack(n_i, axis=0),
        }
        self.logger.info("Full mean-field simulation complete.")
        return output

    def _draw_nu_inits(self,
                       n_runs: int):

        n, m = n_runs, self.lif_mf.n_populations

        # 1. Base array in [0, 1]
        arr = self.rng.random((n, m))

        # 2. For each row, draw random number of entries to replace
        #    Here the number is uniform in {0, 1, ..., m}
        k = self.rng.integers(0, m + 1, size=n)

        # 3. Replace entries
        for i in range(n):
            # choose k[i] random positions without replacement
            if k[i] > 0:
                idx = self.rng.choice(m, size=k[i], replace=False)
                arr[i, idx] = self.rng.random(k[i]) * 10

        return arr


class ReducedMeanFieldStager(_BaseMeanFieldStager):
    """Run reduced mean-field simulations over a grid."""

    def _plot(
            self,
            grid: np.ndarray,
            force_field: np.ndarray,
            path: np.ndarray,
            *args,
            **kwargs
    ):
        from neuro_mod.mean_field.analysis import visualization as viz
        return viz.gen_potential_plot(grid, force_field, path)

    def __init__(self,
                 config: Path | str | bytes,
                 random_seed: int = None,
                 **kwargs):
        """Initialize the reduced mean-field stager.

        Args:
            config: Path, string, or bytes for the YAML config.
            random_seed: Optional random seed.
            **kwargs: Extra parameters forwarded to the base stager.
        """
        super().__init__(config, random_seed, **kwargs)

    def run(self,
            focus_pops: list[int],
            grid_density: float = .5,
            grid_lims: tuple[float, float] | list[tuple[float, float]] = (0., 60.),):
        """Compute reduced mean-field flow on a grid.

        Args:
            focus_pops: Indices of populations to focus on.
            grid_density: Step size for the grid.
            grid_lims: Limits for grid axes.

        Returns:
            Dictionary containing grid, force field, and potentials.
        """
        self.logger.info("Running reduced mean-field simulation.")
        from neuro_mod.mean_field.analysis.logic import integration as ing
        nu_vecs = self._gen_grid_vecs(focus_pops, grid_density, grid_lims)
        mesh, nu_outs = self._get_effective_rates_on_grid(focus_pops, *nu_vecs)
        num_focus = len(focus_pops)
        grid_shape = tuple(len(v) for v in nu_vecs[:num_focus])
        shape = grid_shape + (mesh.shape[-1],)
        grid = mesh.reshape(shape)[..., :2]
        nu_out_grid = nu_outs.reshape(shape)[..., :2]
        force_field = nu_out_grid - grid
        path = ing.get_path_of_min_res(force_field)
        potential, displacement = ing.compute_line_integral_on_path(grid, force_field, path)
        outputs = {
            "potential": potential,
            "displacement": displacement,
            "force_field": force_field,
            "path": path,
            "nu_outs": nu_outs,
            "mesh": mesh
        }
        self.logger.info("Reduced mean-field simulation complete.")
        return outputs

    def _get_effective_rates_on_grid(self, focus_pops, *nu_vecs):
        from itertools import product
        grid = np.array(list(product(*nu_vecs)), dtype=float)  # or use np.float64

        # Evaluate model for each row
        nu_outs = np.empty((len(grid), self.lif_mf.n_populations))
        for i, point in enumerate(grid):
            nu_outs[i] = self.lif_mf.effective_response_function(focus_pops, point)
        return grid, nu_outs

    def _gen_grid_vecs(self,
                       focus_pops: list[int],
                       grid_density: float,
                       grid_lims: tuple[float, float] | list[tuple[float, float]]):
        num_nu_vecs = len(focus_pops)
        if isinstance(grid_lims, tuple):
            _nu_vec = np.arange(*grid_lims, step=grid_density)
            nu_vecs = [_nu_vec] * num_nu_vecs
        else:
            try:
                assert len(grid_lims) == num_nu_vecs
            except AssertionError:
                raise ValueError("grid_lims must be a tuple of length of the focus populations")
            nu_vecs = [np.arange(*lims, step=grid_density) for lims in grid_lims]
        random_vals = self.rng.random((self.lif_mf.n_populations - num_nu_vecs, 1))
        nu_vecs += list(random_vals)
        return nu_vecs
