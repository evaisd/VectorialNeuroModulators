"""Sensitivity matrix estimation from canonical basis sweep simulations.

The sensitivity matrix G has shape (N_attractors, C) where:
    G[S, c] = ∂ log p(S) / ∂ α_c |_{α=0}

estimated as the OLS slope of log p(S, α_c) vs α_c over the linear regime.

Under the linear scoring model (Δ log p(S) ≈ β · δ^T x_S), G ≈ β · X
where X is the k-hot attractor matrix. Deviations from this capture
nonlinear network effects not predicted by the abstract theory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading sweep outputs
# ---------------------------------------------------------------------------

def load_sweep_probabilities(
    sweep_dir: Path | str,
    k_filter: int | None = None,
    min_occurrences: int = 5,
) -> pd.DataFrame:
    """Load per-attractor occurrence probabilities from a completed sweep run.

    Expects the standard pipeline output structure:
        sweep_dir/
          analysis/dataframe.pkl   (SNNAnalyzer-style DataFrame)
          OR
          dataframes/aggregated.csv

    The probability p(S, α) is computed as:
        p(S, α) = total_duration(S, α) / total_window_duration(α)

    Args:
        sweep_dir: Path to the sweep run output directory.
        k_filter: If given, retain only attractors with exactly k active
            clusters (e.g. k_filter=3 for 3-hot attractors).
        min_occurrences: Drop (attractor, sweep_value) pairs with fewer
            than this many observed occurrences (unreliable estimates).

    Returns:
        DataFrame with columns:
            sweep_value (float): perturbation scale α at each sweep step
            clusters (tuple): attractor identity as a tuple of cluster indices
            occurrences (int): number of observed occurrences
            total_duration_ms (float): summed dwell time in ms
            prob (float): estimated steady-state probability p(S, α)
    """
    sweep_dir = Path(sweep_dir)

    # Try pickle first (richer), fall back to CSV
    pkl_path = sweep_dir / "analysis" / "dataframe.pkl"
    csv_path = sweep_dir / "dataframes" / "aggregated.csv"

    if pkl_path.exists():
        df_raw = pd.read_pickle(pkl_path)
    elif csv_path.exists():
        df_raw = pd.read_csv(csv_path)
        # clusters column may be stored as string; convert to tuple
        if df_raw["clusters"].dtype == object and isinstance(
            df_raw["clusters"].iloc[0], str
        ):
            df_raw["clusters"] = df_raw["clusters"].apply(
                lambda s: tuple(int(x) for x in s.strip("()").split(",") if x.strip())
            )
    else:
        raise FileNotFoundError(
            f"No aggregated output found in {sweep_dir}. "
            "Expected analysis/dataframe.pkl or dataframes/aggregated.csv."
        )

    # Normalise column names to expected schema
    required = {"sweep_value", "clusters", "duration"}
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(
            f"Missing columns {missing} in DataFrame loaded from {sweep_dir}. "
            f"Available: {list(df_raw.columns)}"
        )

    # Optional cluster-count filter
    if k_filter is not None:
        df_raw = df_raw[df_raw["clusters"].apply(len) == k_filter].copy()

    # Aggregate per (attractor, sweep_value)
    grouped = (
        df_raw.groupby(["clusters", "sweep_value"])
        .agg(
            occurrences=("duration", "count"),
            total_duration_ms=("duration", "sum"),
        )
        .reset_index()
    )

    # Drop low-count entries
    grouped = grouped[grouped["occurrences"] >= min_occurrences].copy()

    # Compute probability: total_duration / total_window_per_sweep_step
    window_ms = (
        df_raw.groupby("sweep_value")["duration"].sum().rename("window_ms")
    )
    grouped = grouped.join(window_ms, on="sweep_value")
    grouped["prob"] = grouped["total_duration_ms"] / grouped["window_ms"]
    grouped = grouped.drop(columns=["window_ms"])

    return grouped.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Loader for attractors.npy pipeline output
# ---------------------------------------------------------------------------

def load_attractors_from_npy(
    exp_dir: Path | str,
    k_filter: int | None = None,
    min_occurrences: int = 5,
) -> tuple[list[tuple[int, ...]], np.ndarray, dict]:
    """Load attractor landscape from the processed/sweep_0/ pipeline output.

    Reads the ``attractors.npy`` dict produced by the SNN batch processor and
    estimates steady-state probabilities as:

        p(S) = counts(S) * mean_lifespan_ms / total_duration_ms

    where ``mean_lifespan_ms`` is taken from ``dataframes/sweep_summary.parquet``
    if available, otherwise approximated as
    ``total_duration_ms / total_occurrences``.

    Args:
        exp_dir: Path to a single experiment directory (one M / target pair),
            e.g. ``simulations/capacity_validation/M1/0_6_12``.
        k_filter: If given, retain only attractors with exactly k active
            clusters (e.g. k_filter=3 for 3-hot attractors).
        min_occurrences: Drop attractors seen fewer than this many times.

    Returns:
        attractor_tuples: List of attractor identity tuples.
        probabilities:    ndarray of p(S) estimates aligned to attractor_tuples.
        meta:             Dict with keys ``total_duration_ms``, ``n_runs``,
                          ``mean_lifespan_ms``, ``total_occurrences``.

    Raises:
        FileNotFoundError: If ``processed/sweep_0/attractors.npy`` or
            ``processed/sweep_0/processor_config.json`` are missing.
    """
    import json

    exp_dir = Path(exp_dir)
    npy_path = exp_dir / "processed" / "sweep_0" / "attractors.npy"
    cfg_path = exp_dir / "processed" / "sweep_0" / "processor_config.json"

    if not npy_path.exists():
        raise FileNotFoundError(f"attractors.npy not found: {npy_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"processor_config.json not found: {cfg_path}")

    att_dict = np.load(npy_path, allow_pickle=True).item()

    with open(cfg_path) as f:
        cfg = json.load(f)
    total_duration_ms: float = float(cfg["total_duration_ms"])
    n_runs: int = int(cfg["n_runs"])

    # Count occurrences across all attractors
    all_counts = {tup: int(info["#"]) for tup, info in att_dict.items()}
    total_occurrences = sum(all_counts.values())

    # Prefer parquet-derived mean lifespan; fall back to global estimate
    parquet_path = exp_dir / "dataframes" / "sweep_summary.parquet"
    if parquet_path.exists():
        summary = pd.read_parquet(parquet_path)
        mean_lifespan_ms = float(summary["mean_lifespan_ms"].iloc[0])
    else:
        mean_lifespan_ms = total_duration_ms / total_occurrences if total_occurrences > 0 else 1.0

    meta = {
        "total_duration_ms": total_duration_ms,
        "n_runs": n_runs,
        "mean_lifespan_ms": mean_lifespan_ms,
        "total_occurrences": total_occurrences,
    }

    # Build filtered lists
    attractor_tuples: list[tuple[int, ...]] = []
    probabilities: list[float] = []

    for tup, count in all_counts.items():
        if count < min_occurrences:
            continue
        if k_filter is not None and len(tup) != k_filter:
            continue
        p = count * mean_lifespan_ms / total_duration_ms
        attractor_tuples.append(tup)
        probabilities.append(p)

    return attractor_tuples, np.array(probabilities, dtype=np.float64), meta


# ---------------------------------------------------------------------------
# Sensitivity matrix
# ---------------------------------------------------------------------------

def build_G_matrix(
    basis_sweep_dirs: list[Path | str],
    attractor_universe: list[tuple[int, ...]] | None = None,
    alpha_max_linear: float = 0.05,
    k_filter: int | None = None,
    min_occurrences: int = 5,
    min_conditions: int = 10,
) -> tuple[np.ndarray, list[tuple[int, ...]], np.ndarray]:
    """Build the sensitivity matrix G from C canonical basis sweep outputs.

    G[S, c] = OLS slope of log p(S, α_c) vs α_c, restricted to
              α_c ∈ [0, alpha_max_linear].

    Args:
        basis_sweep_dirs: List of C paths, basis_sweep_dirs[c] is the output
            directory for the e_c perturbation (single-cluster rate sweep).
            Must be ordered by cluster index 0, 1, ..., C-1.
        attractor_universe: Canonical attractor list for row ordering.
            If None, derived as the intersection of attractors observed across
            all C sweep conditions (with prob > 0 at α=0 and present in ≥
            min_conditions sweep directories).
        alpha_max_linear: Upper cutoff for the linear-regime fit (default 0.05).
        k_filter: If set, only include k-hot attractors of this size.
        min_occurrences: Passed to load_sweep_probabilities.
        min_conditions: Minimum number of sweep directories in which an
            attractor must appear to be included in the G matrix.

    Returns:
        G:          ndarray of shape (N_attractors, C)
        attractors: list of attractor tuples of length N_attractors (row labels)
        alpha_grid: ndarray of the α values used for fitting (shape (n_steps,))

    Notes:
        Attractors not observed in a given sweep condition get G[S, c] = 0
        (zero sensitivity), which is conservative but may underestimate the
        true sensitivity for rare attractors.
    """
    C = len(basis_sweep_dirs)
    sweep_data: list[pd.DataFrame] = []

    for c, d in enumerate(basis_sweep_dirs):
        df = load_sweep_probabilities(d, k_filter=k_filter, min_occurrences=min_occurrences)
        df["cluster_idx"] = c
        sweep_data.append(df)

    all_data = pd.concat(sweep_data, ignore_index=True)

    # Build attractor universe
    if attractor_universe is None:
        # Keep attractors present in >= min_conditions sweep dirs
        counts = all_data.groupby("clusters")["cluster_idx"].nunique()
        attractor_universe = sorted(counts[counts >= min_conditions].index.tolist())

    # Shared α grid (union of all sweep values, truncated at alpha_max_linear)
    alpha_values = np.sort(
        all_data.loc[all_data["sweep_value"] <= alpha_max_linear, "sweep_value"].unique()
    )

    N = len(attractor_universe)
    G = np.zeros((N, C), dtype=np.float64)
    attractor_index = {a: i for i, a in enumerate(attractor_universe)}

    for c in range(C):
        df_c = sweep_data[c]
        df_c_lin = df_c[df_c["sweep_value"] <= alpha_max_linear].copy()

        for attractor, grp in df_c_lin.groupby("clusters"):
            if attractor not in attractor_index:
                continue
            grp = grp.sort_values("sweep_value")
            alphas = grp["sweep_value"].values
            log_p = np.log(np.clip(grp["prob"].values, 1e-10, None))
            if len(alphas) < 2:
                continue
            # OLS slope via np.polyfit
            slope, _ = np.polyfit(alphas, log_p, 1)
            G[attractor_index[attractor], c] = slope

    return G, list(attractor_universe), alpha_values


# ---------------------------------------------------------------------------
# Linearity validation
# ---------------------------------------------------------------------------

def validate_linearity(
    sweep_probs: pd.DataFrame,
    attractor: tuple[int, ...],
    alpha_max_linear: float = 0.05,
    r2_threshold: float = 0.90,
) -> dict:
    """Test how well log p(S, α) is linear in α up to alpha_max_linear.

    Args:
        sweep_probs: Output of load_sweep_probabilities for a single sweep.
        attractor: Target attractor tuple to test.
        alpha_max_linear: Upper cutoff for the linear regime.
        r2_threshold: Minimum R² to declare linearity.

    Returns:
        dict with keys:
            'attractor':    tuple
            'slope':        float — estimated G[S, c]
            'intercept':    float
            'r2':           float — coefficient of determination
            'is_linear':    bool — R² ≥ r2_threshold
            'alpha_break':  float | None — smallest α where residual > 2σ
            'n_points':     int — number of (α, log p) data points used
    """
    mask = (sweep_probs["clusters"] == attractor) & (
        sweep_probs["sweep_value"] <= alpha_max_linear
    )
    grp = sweep_probs[mask].sort_values("sweep_value")

    if len(grp) < 3:
        return {
            "attractor": attractor,
            "slope": np.nan,
            "intercept": np.nan,
            "r2": np.nan,
            "is_linear": False,
            "alpha_break": None,
            "n_points": len(grp),
        }

    alphas = grp["sweep_value"].values
    log_p = np.log(np.clip(grp["prob"].values, 1e-10, None))

    coeffs = np.polyfit(alphas, log_p, 1)
    slope, intercept = coeffs
    log_p_pred = np.polyval(coeffs, alphas)
    residuals = log_p - log_p_pred

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_p - log_p.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # Find first α where residual exceeds 2 standard deviations
    sigma = np.std(residuals)
    break_mask = np.abs(residuals) > 2 * sigma
    alpha_break = float(alphas[break_mask][0]) if break_mask.any() else None

    return {
        "attractor": attractor,
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "is_linear": bool(r2 >= r2_threshold),
        "alpha_break": alpha_break,
        "n_points": len(grp),
    }


def validate_superposition(
    G: np.ndarray,
    attractors: list[tuple[int, ...]],
    reference_sweep_dir: Path | str,
    reference_delta: np.ndarray,
    p0: np.ndarray,
    alpha_max: float = 0.05,
    k_filter: int | None = None,
) -> dict:
    """Test superposition: compare G-predicted Δlog p to an observed sweep.

    Uses an existing sweep (e.g. snn_rate_sweep_half) as ground truth.
    The predicted log p under perturbation α·reference_delta is:
        log p_pred(S, α) = log p0(S) + α · G[S, :] @ reference_delta

    Args:
        G: Sensitivity matrix, shape (N, C).
        attractors: Row labels for G.
        reference_sweep_dir: Path to an existing sweep output whose perturbation
            direction is reference_delta (e.g. snn_rate_sweep_half).
        reference_delta: ndarray of shape (C,) — the perturbation direction
            used in the reference sweep (need not be unit-norm here).
        p0: ndarray of shape (N,) — baseline probabilities (at α=0).
        alpha_max: Upper cutoff for the comparison.
        k_filter: If set, restrict to k-hot attractors.

    Returns:
        dict with keys:
            'mean_residual':   float — mean |log p_obs - log p_pred| over
                               all (attractor, α) pairs
            'max_residual':    float
            'r2_superposition': float — R² of log p_pred vs log p_obs
            'details':         DataFrame with columns attractor, sweep_value,
                               log_p_obs, log_p_pred, residual
    """
    sweep_probs = load_sweep_probabilities(
        reference_sweep_dir, k_filter=k_filter
    )
    sweep_probs = sweep_probs[sweep_probs["sweep_value"] <= alpha_max]

    attractor_idx = {a: i for i, a in enumerate(attractors)}
    rows = []
    for _, row in sweep_probs.iterrows():
        att = row["clusters"]
        if att not in attractor_idx:
            continue
        i = attractor_idx[att]
        alpha = row["sweep_value"]
        log_p_obs = np.log(max(row["prob"], 1e-10))
        log_p_pred = np.log(max(p0[i], 1e-10)) + alpha * (G[i] @ reference_delta)
        rows.append({
            "attractor": att,
            "sweep_value": alpha,
            "log_p_obs": log_p_obs,
            "log_p_pred": log_p_pred,
            "residual": abs(log_p_obs - log_p_pred),
        })

    if not rows:
        return {
            "mean_residual": np.nan,
            "max_residual": np.nan,
            "r2_superposition": np.nan,
            "details": pd.DataFrame(),
        }

    details = pd.DataFrame(rows)
    obs = details["log_p_obs"].values
    pred = details["log_p_pred"].values
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - obs.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    return {
        "mean_residual": float(details["residual"].mean()),
        "max_residual": float(details["residual"].max()),
        "r2_superposition": float(r2),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_probabilities_from_G(
    G: np.ndarray,
    attractors: list[tuple[int, ...]],
    delta: np.ndarray,
    alpha: float,
    p0: np.ndarray,
    renormalise: bool = True,
) -> np.ndarray:
    """Predict p(S | α·δ) under the linear scoring model using G.

    Model:  log p_pred(S) = log p0(S) + α · G[S, :] @ δ

    Args:
        G: Sensitivity matrix, shape (N, C).
        attractors: Row labels for G (unused in computation, for reference).
        delta: ndarray of shape (C,) — perturbation direction (need not be
            unit-norm; scale is absorbed into α).
        alpha: Perturbation amplitude.
        p0: ndarray of shape (N,) — baseline probabilities at α=0.
        renormalise: If True, normalise the output to sum to 1.

    Returns:
        p_pred: ndarray of shape (N,) — predicted probabilities.
    """
    log_p0 = np.log(np.clip(p0, 1e-10, None))
    log_p_pred = log_p0 + alpha * (G @ delta)
    # Shift to avoid numerical overflow before exp
    log_p_pred -= log_p_pred.max()
    p_pred = np.exp(log_p_pred)
    if renormalise:
        p_pred /= p_pred.sum()
    return p_pred
