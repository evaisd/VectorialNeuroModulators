"""SDP-based subspace optimisation for neuromodulatory controllability.

Implements the Grassmannian optimisation described in raw/notes/capacity/:
  maximise  γ
  s.t.  d_{SS'}^T Π d_{SS'} ≥ γ   for all (S, S') in bottleneck pairs
        0 ⪯ Π ⪯ I,  tr(Π) = M
  variable: Π ∈ S^C (symmetric C×C PSD matrix), γ ∈ R

The optimal Π encodes a rank-M projection subspace W*(M). The optimal
targeting direction for attractor S_0 is δ*_{S0} = Π x_{S0} / ‖Π x_{S0}‖.
"""

from __future__ import annotations

import warnings
import numpy as np

try:
    import cvxpy as cp
    _CVXPY_AVAILABLE = True
except ImportError:
    _CVXPY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Attractor geometry
# ---------------------------------------------------------------------------

def build_attractor_vectors(
    attractor_tuples: list[tuple[int, ...]],
    C: int = 18,
) -> np.ndarray:
    """Convert attractor cluster-tuple identities to k-hot indicator matrix.

    Args:
        attractor_tuples: List of tuples of active cluster indices,
            e.g. [(0, 5, 12), (1, 3, 7), ...].
        C: Number of clusters (ambient dimension).

    Returns:
        X: ndarray of shape (N, C). X[i, j] = 1 if cluster j is active
            in attractor i, else 0.
    """
    N = len(attractor_tuples)
    X = np.zeros((N, C), dtype=np.float64)
    for i, tup in enumerate(attractor_tuples):
        for j in tup:
            X[i, j] = 1.0
    return X


def build_difference_vectors(
    X: np.ndarray,
    vocabulary_indices: list[int],
    k: int = 3,
    bottleneck_only: bool = True,
) -> list[tuple[int, int, np.ndarray]]:
    """Build normalised difference vectors d_{SS'} = (x_S - x_{S'}) / sqrt(k).

    By default only generates bottleneck pairs |S △ S'| = 2 (attractors
    differing by exactly one cluster swap). These are the hardest pairs to
    discriminate and are sufficient to saturate the SDP for the standard
    vocabulary (see raw/notes/capacity/02b_minimal_saturating_set.md).

    Args:
        X: k-hot matrix of shape (N, C) from build_attractor_vectors.
        vocabulary_indices: Row indices of X that form the target vocabulary T.
        k: Number of active clusters per attractor (used for normalisation).
        bottleneck_only: If True, only emit pairs with |S △ S'| = 2.
            If False, emit all ordered pairs in the vocabulary.

    Returns:
        List of (s_idx, s_prime_idx, d_vec) triples where d_vec has shape (C,)
        and indices are positions within vocabulary_indices.
    """
    voc = vocabulary_indices
    pairs = []
    for i, si in enumerate(voc):
        for j, sj in enumerate(voc):
            if i == j:
                continue
            diff = X[si] - X[sj]
            sym_diff_size = int(np.sum(np.abs(diff)))  # |S △ S'|
            if bottleneck_only and sym_diff_size != 2:
                continue
            d = diff / np.sqrt(k)
            pairs.append((i, j, d))
    return pairs


# ---------------------------------------------------------------------------
# SDP solver
# ---------------------------------------------------------------------------

def solve_subspace_sdp(
    difference_vectors: list[tuple[int, int, np.ndarray]],
    M: int,
    C: int = 18,
    solver: str = "SCS",
    verbose: bool = False,
) -> dict:
    """Solve the SDP for the optimal rank-M projection matrix.

    Formulation (maximise minimax targeting margin):
        maximise  γ
        s.t.  d^T Π d ≥ γ   for all (_, _, d) in difference_vectors
              Π ⪰ 0
              I - Π ⪰ 0
              tr(Π) = M
        variable: Π ∈ S^C, γ ∈ R

    The constraint d^T Π d = tr(Π d d^T) is linear in Π, so this is a
    standard SDP. cvxpy with the SCS solver handles C=18 in under 1 s.

    Args:
        difference_vectors: Output of build_difference_vectors.
        M: Rank / trace target (number of neuromodulatory modes).
        C: Ambient dimension (number of clusters, default 18).
        solver: cvxpy solver name. "SCS" is fast; "MOSEK" is more accurate
            if available.
        verbose: Forward verbosity flag to cvxpy solver.

    Returns:
        dict with keys:
            'Pi':     ndarray (C, C) — optimal projection matrix
            'gamma':  float — achieved minimax margin (≥ 0)
            'status': str — solver status string
            'M':      int — trace rank used

    Raises:
        ImportError: if cvxpy is not installed.
        ValueError: if solver returns infeasible or unbounded status.
    """
    if not _CVXPY_AVAILABLE:
        raise ImportError(
            "cvxpy is required for SDP solving. Install with: pip install cvxpy"
        )

    Pi = cp.Variable((C, C), symmetric=True)
    gamma = cp.Variable()

    constraints = [
        Pi >> 0,                       # Π ⪰ 0
        np.eye(C) - Pi >> 0,           # I - Π ⪰ 0  (i.e., Π ⪯ I)
        cp.trace(Pi) == float(M),      # tr(Π) = M
    ]
    for _, _, d in difference_vectors:
        constraints.append(cp.quad_form(d, Pi) >= gamma)

    prob = cp.Problem(cp.Maximize(gamma), constraints)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status in ("infeasible", "unbounded"):
        raise ValueError(
            f"SDP solver returned status '{prob.status}' for M={M}. "
            "Check that difference_vectors is non-empty."
        )
    if prob.status not in ("optimal", "optimal_inaccurate"):
        warnings.warn(
            f"SDP solver status '{prob.status}' for M={M}. "
            "Results may be inaccurate.",
            RuntimeWarning,
            stacklevel=2,
        )

    Pi_val = Pi.value if Pi.value is not None else np.full((C, C), np.nan)
    gamma_val = float(gamma.value) if gamma.value is not None else np.nan

    return {
        "Pi": Pi_val,
        "gamma": gamma_val,
        "status": prob.status,
        "M": M,
    }


# ---------------------------------------------------------------------------
# Targeting directions
# ---------------------------------------------------------------------------

def compute_targeting_direction(
    Pi: np.ndarray,
    x_S0: np.ndarray,
) -> np.ndarray:
    """Compute the optimal unit-norm perturbation direction for target S0.

    δ*_{S0} = Π x_{S0} / ‖Π x_{S0}‖

    Args:
        Pi: ndarray of shape (C, C) — optimal projection from solve_subspace_sdp.
        x_S0: ndarray of shape (C,) — k-hot indicator vector for the target.

    Returns:
        delta_star: ndarray of shape (C,) — unit-norm targeting direction.

    Raises:
        ValueError: if ‖Π x_{S0}‖ is numerically zero (attractor not in subspace).
    """
    projected = Pi @ x_S0
    norm = np.linalg.norm(projected)
    if norm < 1e-10:
        raise ValueError(
            "‖Π x_{S0}‖ ≈ 0: the target attractor lies outside the optimal "
            "subspace. This can happen when the target is not in the vocabulary "
            "used to solve the SDP."
        )
    return projected / norm


# ---------------------------------------------------------------------------
# Capacity curve
# ---------------------------------------------------------------------------

def capacity_curve(
    vocabulary: list[tuple[int, ...]],
    M_range: list[int] | range,
    C: int = 18,
    k: int = 3,
    solver: str = "SCS",
    verbose: bool = False,
) -> dict:
    """Compute the SDP optimal margin Γ_min(W*; M) across a range of M.

    Args:
        vocabulary: Target attractor tuples (the vocabulary T).
            For full saturation at C=18, k=3, use ≥ 12 attractors from
            build_minimal_saturating_vocabulary.
        M_range: Values of M to evaluate, e.g. range(1, 19).
        C: Number of clusters.
        k: Active clusters per attractor (for normalisation).
        solver: cvxpy solver.
        verbose: Solver verbosity.

    Returns:
        dict with keys:
            'M_values':    list[int]
            'gamma_opt':   list[float] — Γ_min(W*; M) for each M
            'Pi_matrices': list[ndarray] — optimal Π for each M
            'statuses':    list[str] — solver status per M
            'is_isotropic': list[bool] — True if Π ≈ (M/C)·I within 1e-2
    """
    M_range = list(M_range)
    X = build_attractor_vectors(vocabulary, C=C)
    voc_idx = list(range(len(vocabulary)))
    diff_vecs = build_difference_vectors(X, voc_idx, k=k, bottleneck_only=True)

    gammas, Pi_mats, statuses, is_iso = [], [], [], []

    for M in M_range:
        result = solve_subspace_sdp(diff_vecs, M=M, C=C, solver=solver, verbose=verbose)
        gammas.append(result["gamma"])
        Pi_mats.append(result["Pi"])
        statuses.append(result["status"])

        # Permutation symmetry conjecture: Π* ≈ (M/C) I
        Pi = result["Pi"]
        if Pi is not None and not np.any(np.isnan(Pi)):
            iso_pred = (M / C) * np.eye(C)
            is_iso.append(bool(np.allclose(Pi, iso_pred, atol=1e-2)))
        else:
            is_iso.append(False)

    return {
        "M_values": M_range,
        "gamma_opt": gammas,
        "Pi_matrices": Pi_mats,
        "statuses": statuses,
        "is_isotropic": is_iso,
    }


# ---------------------------------------------------------------------------
# Theoretical vocabulary
# ---------------------------------------------------------------------------

def build_minimal_saturating_vocabulary(
    C: int = 18,
    k: int = 3,
) -> list[tuple[int, ...]]:
    """Construct a minimal saturating vocabulary of size ceil(2C/k).

    Strategy: two disjoint uniform partitions of [0..C-1] into k-sized groups.
      Partition 1: (0,1,2), (3,4,5), ..., (C-k, C-k+1, ..., C-1)
      Partition 2: cyclic shift by 1 — (1,2,3), (4,5,6), ..., (C-1,0,1)

    This guarantees every ordered cluster pair (i→j) is covered by some
    attractor S with i ∈ S and j ∉ S, saturating all directed bottleneck
    constraints in the SDP (see raw/notes/capacity/02b_minimal_saturating_set.md).

    Args:
        C: Number of clusters (must be divisible by k for exact construction;
            otherwise the last group is padded to the next attractor boundary).
        k: Active clusters per attractor.

    Returns:
        List of ceil(2C/k) tuples, each of length k.

    Raises:
        ValueError: if C < 2*k (too few clusters to form two disjoint partitions).
    """
    if C < 2 * k:
        raise ValueError(f"C={C} must be ≥ 2k={2*k} for two disjoint partitions.")

    clusters = list(range(C))
    vocabulary = []

    # Partition 1: non-overlapping groups of k
    for start in range(0, C, k):
        group = tuple(clusters[start:start + k])
        if len(group) == k:
            vocabulary.append(group)

    # Partition 2: cyclic shift by 1
    shifted = clusters[1:] + clusters[:1]
    for start in range(0, C, k):
        group = tuple(shifted[start:start + k])
        if len(group) == k:
            vocabulary.append(group)

    return vocabulary
