"""Core mean-field equations and stability utilities."""

from typing import Callable
import numpy as np


__all__ = [
    'equations',
    'integrand',
    'determine_stability',
    'get_jacobian',
    'brunei_sergei'
]


def equations(
        nu: np.ndarray,
        a_mat: np.ndarray,
        b_mat: np.ndarray,
        ext_mu: np.ndarray,
        ext_sigma: np.ndarray,    
):
    """Compute mean and standard deviation of input currents.

    Args:
        nu: Population firing rates.
        a_mat: Effective coupling matrix for the mean input.
        b_mat: Effective coupling matrix for the variance.
        ext_mu: External mean input per population.
        ext_sigma: External variance input per population.

    Returns:
        Tuple `(mu, sigma)` for the mean and standard deviation inputs.
    """
    mu = (a_mat @ nu + ext_mu)
    var = b_mat @ nu + ext_sigma
    sigma = np.sqrt(np.maximum(var, 1e-12) / 1.)
    return mu, sigma


def determine_stability(
        nu_star: np.ndarray,
        f: Callable[[np.ndarray], list[float]],
):
    """Classify stability of a fixed point using Jacobian eigenvalues.

    Args:
        nu_star: Fixed point rates.
        f: Function returning residuals of the fixed-point equations.

    Returns:
        Tuple `(fp_type, eigvals)` where `fp_type` is a descriptive string
        and `eigvals` are the Jacobian eigenvalues.
    """
    tol = 1e-6
    jacobian = get_jacobian(
        nu_star,
        f,
    )

    # Check for NaN/Inf in Jacobian
    if not np.all(np.isfinite(jacobian)):
        return "undefined (numerical issues)", np.array([np.nan])

    eigvals = np.linalg.eig(jacobian)[0]
    real_parts = eigvals.real

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


def get_jacobian(
        nu_star: np.ndarray,
        f: Callable[[np.ndarray], list[float]],
        eps: float = 1e-6,
):
    """Estimate the Jacobian matrix via finite differences.

    Args:
        nu_star: Point at which to evaluate the Jacobian.
        f: Function returning residuals.
        eps: Perturbation size for finite differences.

    Returns:
        The Jacobian matrix of `f` at `nu_star`, with sign flipped to match
        `d(dot{nu})/dnu`.
    """
    f0 = np.array(f(nu_star))
    n = len(nu_star)
    jacobian = np.zeros((n, n))
    pert = np.zeros_like(nu_star)
    for i in range(n):
        pert[i] = eps
        f1 = np.array(f(nu_star + pert))
        jacobian[:, i] = (f1 - f0) / eps
    return -jacobian  # minus sign for d(dot{nu})/dnu


def integrand(z):
    """Integrand for the LIF transfer function expression."""
    from scipy.special import erfcx
    if z < -15:
        return (1 - 1 / (2 * z ** 2) + 3 / (4 * z ** 4) - 15 / (8 * z ** 6)) * (-1 / z)

    else:
        return np.sqrt(np.pi) * erfcx(-z)


def brunei_sergei(tau_s: np.ndarray | float, tau_m: np.ndarray | float):
    """Compute the Brunel-Sergei correction term.

    Args:
        tau_s: Synaptic time constant(s).
        tau_m: Membrane time constant(s).

    Returns:
        Correction term `b_s` used in LIF response functions.
    """
    from scipy.special import zeta
    a = -zeta(1 / 2) / np.sqrt(2)
    b_s = a * np.sqrt(tau_s / tau_m)
    return b_s
