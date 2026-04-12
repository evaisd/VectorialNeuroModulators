"""Leakage characterisation for targeted neuromodulatory perturbations.

Given a validation sweep (one targeting direction δ*_{S0}(M), swept over
scales α), this module extracts:
  - p(S | α) for the target S_0 and all competitors
  - The role of each competitor (bottleneck vs. distant)
  - The metastable boundary: largest α where dynamics remain metastable
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from neuro_mod.analysis.capacity.sensitivity import load_sweep_probabilities


# ---------------------------------------------------------------------------
# Per-attractor role classification
# ---------------------------------------------------------------------------

def classify_attractor_role(
    attractor: tuple[int, ...],
    target: tuple[int, ...],
    delta_star: np.ndarray,
    target_x: np.ndarray,
) -> dict:
    """Classify an attractor relative to a target and perturbation direction.

    Args:
        attractor: The attractor to classify.
        target: The targeted attractor S_0.
        delta_star: ndarray of shape (C,) — the unit-norm targeting direction.
        target_x: ndarray of shape (C,) — k-hot indicator for the target.

    Returns:
        dict with keys:
            'overlap':          int — |S ∩ target| (number of shared clusters)
            'symmetric_diff':   int — |S △ target|
            'role':             str — 'target', 'bottleneck', 'partial', 'orthogonal'
            'linear_score':     float — δ*^T x_S (raw perturbation score)
            'score_contrast':   float — δ*^T x_S - δ*^T x_{S0} (relative to target)
            'direction':        str — 'boost', 'suppress', or 'neutral'
    """
    if attractor == target:
        x_S = target_x
        return {
            "overlap": len(target),
            "symmetric_diff": 0,
            "role": "target",
            "linear_score": float(delta_star @ x_S),
            "score_contrast": 0.0,
            "direction": "boost",
        }

    x_S = np.zeros(len(delta_star))
    for c in attractor:
        x_S[c] = 1.0

    overlap = len(set(attractor) & set(target))
    sym_diff = len(set(attractor) ^ set(target))

    if sym_diff == 2:
        role = "bottleneck"
    elif 0 < overlap < len(target):
        role = "partial"
    else:
        role = "orthogonal"

    target_score = float(delta_star @ target_x)
    linear_score = float(delta_star @ x_S)
    contrast = linear_score - target_score

    if contrast > 0.05:
        direction = "boost"
    elif contrast < -0.05:
        direction = "suppress"
    else:
        direction = "neutral"

    return {
        "overlap": overlap,
        "symmetric_diff": sym_diff,
        "role": role,
        "linear_score": linear_score,
        "score_contrast": contrast,
        "direction": direction,
    }


# ---------------------------------------------------------------------------
# Full leakage profile
# ---------------------------------------------------------------------------

def compute_leakage_profile(
    sweep_dir: Path | str,
    target: tuple[int, ...],
    delta_star: np.ndarray,
    vocabulary: list[tuple[int, ...]] | None = None,
    G: np.ndarray | None = None,
    attractor_universe: list[tuple[int, ...]] | None = None,
    p0: np.ndarray | None = None,
    k_filter: int | None = None,
    min_occurrences: int = 5,
) -> pd.DataFrame:
    """Compute p(S | α) for all attractors in a validation sweep.

    Args:
        sweep_dir: Path to the validation sweep output directory.
        target: The targeted attractor S_0.
        delta_star: ndarray of shape (C,) — the perturbation direction used.
        vocabulary: If given, used for competitor classification.
        G: If given, overlay theoretical predictions via the linear model.
        attractor_universe: All known attractors (for normalisation).
            If None, derived from the sweep data.
        p0: Baseline probabilities aligned to attractor_universe.
            Required if G is given.
        k_filter: If set, restrict to k-hot attractors.
        min_occurrences: Passed to load_sweep_probabilities.

    Returns:
        DataFrame with columns:
            alpha               float — perturbation scale
            attractor           tuple — attractor identity
            prob                float — empirical probability p(S | α)
            prob_pred           float | NaN — predicted probability (if G given)
            role                str — target / bottleneck / partial / orthogonal
            linear_score        float — δ*^T x_S
            score_contrast      float — δ*^T x_S - δ*^T x_{S0}
            in_vocabulary       bool — True if attractor is in T
    """
    sweep_probs = load_sweep_probabilities(
        sweep_dir, k_filter=k_filter, min_occurrences=min_occurrences
    )

    C = len(delta_star)
    target_x = np.zeros(C)
    for c in target:
        target_x[c] = 1.0

    vocabulary_set = set(vocabulary) if vocabulary is not None else set()

    # Build per-attractor role classifications
    all_attractors = sweep_probs["clusters"].unique().tolist()
    role_map = {
        att: classify_attractor_role(att, target, delta_star, target_x)
        for att in all_attractors
    }

    # Optional: predictions from G
    pred_map: dict[tuple, float] = {}
    if G is not None and p0 is not None and attractor_universe is not None:
        from neuro_mod.analysis.capacity.sensitivity import predict_probabilities_from_G
        att_idx = {a: i for i, a in enumerate(attractor_universe)}
        alpha_values = sorted(sweep_probs["sweep_value"].unique())
        for alpha in alpha_values:
            p_pred = predict_probabilities_from_G(G, attractor_universe, delta_star, alpha, p0)
            for att, i in att_idx.items():
                pred_map[(att, alpha)] = float(p_pred[i])

    rows = []
    for _, row in sweep_probs.iterrows():
        att = row["clusters"]
        alpha = row["sweep_value"]
        role_info = role_map.get(att, {})
        prob_pred = pred_map.get((att, alpha), np.nan)
        rows.append({
            "alpha": alpha,
            "attractor": att,
            "prob": row["prob"],
            "prob_pred": prob_pred,
            "role": role_info.get("role", "unknown"),
            "linear_score": role_info.get("linear_score", np.nan),
            "score_contrast": role_info.get("score_contrast", np.nan),
            "in_vocabulary": att in vocabulary_set,
        })

    return pd.DataFrame(rows).sort_values(["alpha", "role"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metastable boundary detection
# ---------------------------------------------------------------------------

def find_metastable_boundary(
    leakage_df: pd.DataFrame,
    target: tuple[int, ...],
    criteria: list[str] | None = None,
    k: int | None = None,
    n_states_threshold: int = 5,
) -> dict:
    """Identify the α at which metastable dynamics become degenerate.

    Three complementary criteria:

    'target_dominates':
        Largest α where p(S_0) strictly exceeds all competitors.
        Signals the onset of selective targeting.

    'total_k_fraction':
        Largest α where the total probability mass in k-hot attractors
        remains > 0.5. Below this the network collapses to a single dominant
        state (driven regime). Requires k argument.

    'num_states':
        Largest α where the number of attractors with p > 0.01 is ≥
        n_states_threshold. Captures landscape narrowing.

    Args:
        leakage_df: Output of compute_leakage_profile.
        target: Target attractor S_0.
        criteria: List of criterion names to compute.
            Defaults to all three.
        k: Active cluster count, required for 'total_k_fraction'.
        n_states_threshold: Threshold for 'num_states' criterion.

    Returns:
        dict mapping criterion_name -> alpha_boundary (float | None).
        None means the criterion was never violated in the observed range.
    """
    if criteria is None:
        criteria = ["target_dominates", "total_k_fraction", "num_states"]

    alpha_values = sorted(leakage_df["alpha"].unique())
    result = {}

    if "target_dominates" in criteria:
        boundary = None
        for alpha in alpha_values:
            at_alpha = leakage_df[leakage_df["alpha"] == alpha]
            p_target = at_alpha.loc[
                at_alpha["attractor"] == target, "prob"
            ].values
            if len(p_target) == 0:
                continue
            p_target = p_target[0]
            p_others = at_alpha.loc[
                at_alpha["attractor"] != target, "prob"
            ].values
            if len(p_others) > 0 and p_target > p_others.max():
                boundary = alpha
        result["target_dominates"] = boundary

    if "total_k_fraction" in criteria:
        if k is None:
            result["total_k_fraction"] = None
        else:
            boundary = None
            for alpha in alpha_values:
                at_alpha = leakage_df[leakage_df["alpha"] == alpha]
                k_hot = at_alpha[at_alpha["attractor"].apply(len) == k]
                total_k = k_hot["prob"].sum()
                if total_k >= 0.5:
                    boundary = alpha
            result["total_k_fraction"] = boundary

    if "num_states" in criteria:
        boundary = None
        for alpha in alpha_values:
            at_alpha = leakage_df[leakage_df["alpha"] == alpha]
            n_active = (at_alpha["prob"] > 0.01).sum()
            if n_active >= n_states_threshold:
                boundary = alpha
        result["num_states"] = boundary

    return result
