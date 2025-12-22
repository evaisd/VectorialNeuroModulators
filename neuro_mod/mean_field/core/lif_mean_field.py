
import numpy as np


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

    def __init__(
            self,
            n_clusters: int,
            c_matrix: np.ndarray[np.ndarray[int]],
            j_matrix: np.ndarray[np.ndarray[float]],
            j_ext: np.ndarray[float],
            c_ext: np.ndarray[int],
            nu_ext: np.ndarray[float],
            tau_membrane: np.ndarray[float] | list[float] | float,
            tau_synaptic: np.ndarray[float] | list[float] | float,
            threshold: float | list[float] | np.ndarray[float],
            reset_voltage: float | list[float] | np.ndarray[float],
            tau_refractory: float = 0.,
            *args,
            **kwargs
    ):
        self.n_clusters = int(n_clusters)
        self.n_populations = j_matrix.shape[0]
        self.j_mat = np.asarray(j_matrix, dtype=float)
        self.c_mat = np.asarray(c_matrix, dtype=np.uint16)
        shape = self.c_mat.shape
        c_ext = self._gen_ext_arrs(np.asarray(c_ext))
        j_ext = self._gen_ext_arrs(np.asarray(j_ext))
        nu_ext = self._gen_ext_arrs(np.asarray(nu_ext))

        self.tau_m = self._gen_ext_arrs(np.asarray(tau_membrane))
        self.tau_s = self._gen_ext_arrs(np.asarray(tau_synaptic))
        self.tau_ref = self._gen_ext_arrs(np.asarray(tau_refractory))
        self.threshold = self._gen_ext_arrs(np.asarray(threshold))
        self.reset_potential = self._gen_ext_arrs(np.asarray(reset_voltage))

        self.ext_mu = j_ext * c_ext * self.tau_m * nu_ext
        self.ext_sigma = j_ext * j_ext * c_ext * self.tau_m * nu_ext
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
        mu = (self.a_mat @ nu_p + self.ext_mu)[self._dynamic_pops]
        var = self.b_mat @ nu_p + self.ext_sigma
        sigma = np.sqrt(np.maximum(var, 1e-12) / 1.)[self._dynamic_pops]
        return mu, sigma

    def response_function(
            self,
            mu: np.ndarray,
            sigma: np.ndarray,
            **params) -> np.ndarray:
        """Compute firing rates from input statistics using the LIF transfer.

        Args:
            mu: Mean input currents for each population.
            sigma: Standard deviation of input currents for each population.
            **params: Additional keyword arguments (currently unused).

        Returns:
            Array of firing rates for each population.
        """

        from scipy import integrate

        mu = np.asarray(mu, float)
        sigma = np.maximum(np.asarray(sigma, float), 1e-12)
        rates = np.empty_like(mu, float)

        b_s = self._brunel_sergei()
        for i in range(mu.size):
            tau_m = self.tau_m[self._dynamic_pops][i]
            alpha = (self.threshold[self._dynamic_pops][i] - mu[i]) / sigma[i]
            alpha += b_s[i]
            beta = (self.reset_potential[self._dynamic_pops][i] - mu[i]) / sigma[i]
            beta += b_s[i]

            # integrate from β (reset) to α (threshold)
            val, _ = integrate.quad(self._integrand, beta, alpha,
                                    epsabs=1e-12, epsrel=1e-12)
            denom = self.tau_ref[i] + tau_m * val
            rates[i] = 1.0 / max(denom, 1e-12)

        return rates

    def solve_rates(
            self,
            nu_init: np.ndarray | None = None,
    ):
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
        """Classify the linear stability of a fixed point.

        Args:
            nu_star: Fixed-point firing rates for all populations.

        Returns:
            Tuple `(fp_type, eigvals)` where:

            * `fp_type` is a human-readable string describing the fixed-point
              type (e.g. ``"stable node"`` or ``"saddle"``).
            * `eigvals` are the eigenvalues of the Jacobian at the fixed point.
        """
        tol = 1e-6
        jacobian = self._get_jacobian(nu_star)
        eigvals = np.linalg.eig(jacobian)[0]
        real_parts = eigvals.real

        # Count eigenvalue signs
        n_pos = np.sum(real_parts > tol)
        n_neg = np.sum(real_parts < -tol)
        n_zero = len(eigvals) - n_pos - n_neg

        if n_pos == 0 and n_zero == 0:
            fp_type = "stable node"
        elif n_neg == 0 and n_zero == 0:
            fp_type = "unstable node"
        elif n_pos > 0 and n_neg > 0:
            fp_type = "saddle"
        elif np.any(np.abs(eigvals.imag) > tol):
            if np.all(real_parts < -tol):
                fp_type = "stable focus (spiral sink)"
            elif np.all(real_parts > tol):
                fp_type = "unstable focus (spiral source)"
            else:
                fp_type = "spiral saddle"
        else:
            fp_type = "neutral / marginal"

        return fp_type, eigvals

    def _get_jacobian(self, nu_star: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        f0 = np.array(self._fixed_point_residual(nu_star))
        n = len(nu_star)
        J = np.zeros((n, n))
        pert = np.zeros_like(nu_star)
        for i in range(n):
            pert[i] = eps
            f1 = np.array(self._fixed_point_residual(nu_star + pert))
            J[:, i] = (f1 - f0) / eps
        return -J  # minus sign for d(dot{nu})/dnu

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

    def _brunel_sergei(self):
        from scipy.special import zeta
        a = -zeta(1 / 2) / np.sqrt(2)
        b_s = a * np.sqrt(self.tau_s / self.tau_m)
        return b_s

    @staticmethod
    def _integrand(z):
        from scipy.special import erfcx
        if z < -15:
            return (1 - 1 / (2 * z ** 2) + 3 / (4 * z ** 4) - 15 / (8 * z ** 6)) * (-1 / z)

        else:
            return np.sqrt(np.pi) * erfcx(-z)

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
