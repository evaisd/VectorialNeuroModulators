
import numpy as np


class LIFMeanField:

    def __init__(
            self,
            n_clusters: int,
            c_matrix: np.ndarray,
            j_matrix: np.ndarray,
            ext_mu: np.ndarray,
            ext_sigma: np.ndarray,
            tau_membrane: np.ndarray | None,
            tau_synaptic: np.ndarray | None,
            threshold: float | list[float],
            reset_voltage: float | list[float],
            tau_refractory: float = 0.,
            *args,
            **kwargs
    ):
        self.n_populations = int(n_clusters)
        self.J_matrix = np.asarray(j_matrix, dtype=float)
        self.C_matrix = self._verify_att_shape(c_matrix,
                                               (self.n_populations, self.n_populations),
                                               "C Matrix")
        self.J_matrix = self._verify_att_shape(j_matrix,
                                               (self.n_populations, self.n_populations),
                                               "J Matrix")
        self.ext_mu = np.asarray(ext_mu, dtype=float)
        self.ext_sigma = np.asarray(ext_sigma, dtype=float)
        self.ext_mu = self._verify_att_shape(ext_mu,
                                             (self.n_populations,),
                                             "External mu")
        self.ext_sigma = self._verify_att_shape(ext_sigma,
                                                (self.n_populations,),
                                                "External sigma")
        self.tau_m = self._verify_att_shape(tau_membrane, (self.n_populations,), "tau_m")

        self.tau_s = self._verify_att_shape(tau_synaptic, (self.n_populations,), "tau_s")
        self.tau_ref = self._verify_att_shape(tau_refractory, (self.n_populations,), "tau_ref")
        self.threshold = self._verify_att_shape(threshold, (self.n_populations,), "V threshold")
        self.reset_potential = self._verify_att_shape(reset_voltage, (self.n_populations,), "V reset")

        self.A_mat = self.tau_m * self.C_matrix * self.J_matrix
        self.B_mat = self.tau_m * self.C_matrix * (self.J_matrix ** 2)

        self._dynamic_pops = np.array(range(self.n_populations))
        self._ambient_pops = np.array([])
        self._nu = np.zeros(self.n_populations)

    def equations(self, nu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nu_p = np.zeros(self.n_populations)
        nu_p[self._dynamic_pops] = nu
        if self._ambient_pops.size > 0:
            nu_p[self._ambient_pops] = self._nu[self._ambient_pops]
        mu = (self.A_mat @ nu_p + self.ext_mu)[self._dynamic_pops]
        var = (1 + self.delta) * (self.B_mat @ nu_p + self.ext_sigma)
        sigma = np.sqrt(np.maximum(var, 1e-12) / 1.)[self._dynamic_pops]
        return mu, sigma

    def response_function(
            self,
            mu: np.ndarray,
            sigma: np.ndarray,
            **params) -> np.ndarray:

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
            nu_init: np.ndarray = None,
    ):
        from scipy.optimize import root
        nu_init = np.zeros(self.n_populations) if nu_init is None else nu_init
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

    @staticmethod
    def _verify_att_shape(att, att_shape: tuple[int, ...], name: str):
        att = np.asarray(att)
        if att.shape == att_shape:
            return att
        if att.shape == ():
            return att * np.ones(att_shape, dtype=float)
        else:
            raise ValueError(f"{name} must be scalar or {att_shape}D.")
