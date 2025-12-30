"""Mean-field LIF model implementation for clustered networks."""

import numpy as np
from neuro_mod.mean_field.core import logic
from neuro_mod.mean_field import helpers


class LIFMeanField:
    """Mean-field model of a clustered LIF (leaky integrate-and-fire) network.

    This class implements population-rate dynamics for a recurrent network with
    clustered structure, together with utilities for computing fixed points and
    their linear stability.

    Args:
        n_clusters: Number of populations (clusters) in the network.
        c_matrix: Connectivity matrix containing the number of synapses from
            each presynaptic to each postsynaptic population, shape
            `(n_clusters, n_clusters)`.
        j_matrix: Synaptic efficacy matrix (PSC increments per spike), shape
            `(n_clusters, n_clusters)`.
        j_ext: External synaptic efficacies per population.
        c_ext: External in-degrees per population (number of external synapses).
        nu_ext: External Poisson input rates per population (Hz).
        tau_membrane: Membrane time constants for each population (seconds).
        tau_synaptic: Synaptic time constants for each population (seconds).
        threshold: Firing thresholds for each population.
        reset_voltage: Reset voltages after a spike for each population.
        tau_refractory: Absolute refractory period for each population
            (seconds). Defaults to `0.`.
        *args: Additional positional arguments (ignored, for compatibility).
        **kwargs: Additional keyword arguments (ignored, for compatibility).
    """

    j_mat: np.ndarray | list
    c_mat: np.ndarray | list
    j_ext: np.ndarray | list
    c_ext: np.ndarray | list
    nu_ext: np.ndarray | list
    tau_m: np.ndarray | list
    tau_s: np.ndarray | list
    tau_ref: np.ndarray | list
    reset_potential: float | list
    threshold: float | list | np.ndarray

    def __init__(
            self,
            n_clusters: int,
            c_matrix: np.ndarray,
            j_matrix: np.ndarray,
            j_ext: np.ndarray,
            c_ext: np.ndarray,
            nu_ext: np.ndarray,
            tau_membrane: np.ndarray | list | float,
            tau_synaptic: np.ndarray | list[float] | float,
            threshold: float | list | np.ndarray,
            reset_voltage: float | list | np.ndarray,
            tau_refractory: float = 0.,
            *args,
            **kwargs
    ):
        """Initialize the mean-field model parameters.

        Args:
            n_clusters: Number of clusters.
            c_matrix: Connectivity matrix.
            j_matrix: Synaptic efficacy matrix.
            j_ext: External synaptic efficacies.
            c_ext: External in-degrees.
            nu_ext: External input rates.
            tau_membrane: Membrane time constants.
            tau_synaptic: Synaptic time constants.
            threshold: Firing thresholds.
            reset_voltage: Reset voltages.
            tau_refractory: Refractory period in seconds.
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.
        """
        self.n_clusters = int(n_clusters)
        self.n_populations = j_matrix.shape[0]
        self.j_mat = np.asarray(j_matrix, dtype=float)
        self.c_mat = np.asarray(c_matrix, dtype=np.uint16)

        _project_vars = {
            "tau_m": tau_membrane,
            "tau_s": tau_synaptic,
            "tau_ref": tau_refractory,
            "threshold": threshold,
            "reset_potential": reset_voltage,
            "c_ext": c_ext,
            "j_ext": j_ext,
            "nu_ext": nu_ext,
        }

        for key, value in _project_vars.items():
            value = helpers.gen_ext_arrays(
                value,
                self.n_populations,
                self.n_clusters
            )
            setattr(self, key, value)

        self.ext_mu = self.j_ext * self.c_ext * self.tau_m * self.nu_ext
        self.ext_sigma = self.j_ext * self.j_ext * self.c_ext * self.tau_m * self.nu_ext
        self.a_mat = self.tau_m * self.c_mat * self.j_mat
        self.b_mat = self.tau_m * self.c_mat * (self.j_mat ** 2)

        self._dynamic_pops = np.array(range(self.n_populations))
        self._ambient_pops = np.array([])
        self._nu = np.zeros(self.n_populations)

    def equations(self, nu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute mean and standard deviation of input given population rates.

        Args:
            nu: Firing rates of dynamic populations, shape `(n_dynamic,)`.

        Returns:
            Tuple `(mu, sigma)` where:

            * `mu` is the mean input current to each dynamic population.
            * `sigma` is the standard deviation of the input.
        """
        nu_p = np.zeros(self.n_populations)
        nu_p[self._dynamic_pops] = nu
        if self._ambient_pops.size > 0:
            nu_p[self._ambient_pops] = self._nu[self._ambient_pops]
        mu = (self.a_mat @ nu_p + self.ext_mu)
        var = self.b_mat @ nu_p + self.ext_sigma
        sigma = np.sqrt(np.maximum(var, 1e-12) / 1.)
        return mu[self._dynamic_pops], sigma[self._dynamic_pops]

    def response_function(
            self,
            mu: np.ndarray,
            sigma: np.ndarray
    ) -> np.ndarray:
        """Compute firing rates from input statistics using the LIF transfer.

        Args:
            mu: Mean input currents for each population.
            sigma: Standard deviation of input currents for each population.

        Returns:
            Array of firing rates for each population.
        """

        from scipy import integrate

        mu = np.asarray(mu, float)
        sigma = np.maximum(np.asarray(sigma, float), 1e-12)
        rates = np.empty_like(mu, float)

        b_s = logic.brunei_sergei(tau_m=self.tau_m, tau_s=self.tau_s)
        for i in range(mu.size):
            tau_m = self.tau_m[self._dynamic_pops][i]
            alpha = (self.threshold[self._dynamic_pops][i] - mu[i]) / sigma[i]
            alpha += b_s[i]
            beta = (self.reset_potential[self._dynamic_pops][i] - mu[i]) / sigma[i]
            beta += b_s[i]

            val, _ = integrate.quad(logic.integrand, beta, alpha,
                                    epsabs=1e-12, epsrel=1e-12)
            denom = self.tau_ref[i] + tau_m * val
            rates[i] = 1.0 / max(denom, 1e-12)

        return rates

    def solve_rates(
            self,
            nu_init: np.ndarray | None = None,
    ):
        """Solve for fixed-point firing rates.

        Args:
            nu_init: Optional initial guess for rates.

        Returns:
            SciPy optimization result from `scipy.optimize.root`.
        """
        from scipy.optimize import root
        nu_init = np.zeros(self.n_populations) if nu_init is None else np.asarray(nu_init, float)
        self._nu = nu_init
        # residuals = self._fixed_point_residual(nu_init)
        sol = root(
            self._fixed_point_residual,
            self._nu[self._dynamic_pops],
            jac=False,
            tol=1e-12,
            options={'xtol': 1e-12}
        )
        return sol

    def determine_stability(self, nu_star: np.ndarray):
        """Determine stability of a fixed point.

        Args:
            nu_star: Fixed-point rates to evaluate.

        Returns:
            Tuple `(fp_type, eigvals)` as returned by
            `neuro_mod.mean_field.core.logic.determine_stability`.
        """
        return logic.determine_stability(nu_star, self._fixed_point_residual)

    def _fixed_point_residual(
            self,
            nu_array: np.ndarray,
    ) -> list[float]:
        # self.equations should be modified to return mu and var (σ²)
        # Or you can just recalculate it here. Let's assume it returns mu and var.

        # In MFTEquations.equations:
        #   var = self.B_mat @ nu + self.ext_sigma
        #   return mu, var # Instead of returning sqrt(var)
        mu, sigma = self.equations(nu_array)
        residuals = nu_array - self.response_function(mu, sigma)
        return residuals.tolist()

    def effective_response_function(
            self,
            focus_pops: list[int],
            nu_init: np.ndarray = None,
    ):
        """Compute effective response for a subset of populations.

        Args:
            focus_pops: Population indices to treat as dynamic.
            nu_init: Initial rates for all populations.

        Returns:
            Effective response rates for the focus populations.
        """
        if nu_init is None:
            nu_init = np.zeros(self.n_populations)
        nu_bar = nu_init.copy()
        ambient_pops = [i for i in range(self.n_populations) if i not in focus_pops]
        self._set_dynamic_pops(ambient_pops)
        sol_ambient = self.solve_rates(nu_init)
        nu_bar[ambient_pops] = sol_ambient.x
        # self._set_dynamic_pops(focus_pops)
        self._reset_dynamic_pops()
        mu, sigma = self.equations(nu_bar)
        phi_eff_0 = self.response_function(mu, sigma)
        self._reset_dynamic_pops()
        return phi_eff_0

    def _set_dynamic_pops(self, dynamic_pops: list[int]):
        self._dynamic_pops = np.array(dynamic_pops)
        self._ambient_pops = np.array([i for i in range(self.n_populations) if i not in dynamic_pops])

    def _reset_dynamic_pops(self):
        self._dynamic_pops = np.array(range(self.n_populations))
        self._ambient_pops = np.array([])

    def _gen_ext_arrs(self, original: np.ndarray | list) -> np.ndarray:
        original = np.asarray(original)
        if original.shape == (self.n_populations,):
            return original
        if original.ndim == 0:
            return np.repeat(original, self.n_populations)
        arr = np.zeros(self.n_populations)
        arr[:self.n_clusters + 1] = original[0]
        arr[self.n_clusters + 1:] = original[1]
        return arr
