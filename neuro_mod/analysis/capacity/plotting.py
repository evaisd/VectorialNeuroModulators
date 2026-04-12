"""Figures for capacity analysis: leakage curves, sensitivity heatmap, capacity curve."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def _require_mpl() -> None:
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plotting.")


# ---------------------------------------------------------------------------
# Leakage curves
# ---------------------------------------------------------------------------

def plot_leakage_curves(
    leakage_df: pd.DataFrame,
    target: tuple[int, ...],
    delta_star: np.ndarray | None = None,
    metastable_boundaries: dict | None = None,
    output_path: Path | str | None = None,
    top_n_competitors: int = 5,
) -> "plt.Figure":
    """Four-panel figure showing p(S|α) for target and top competitors.

    Panels:
        1 (top-left):  p(S | α) vs α — linear scale. Target bold, competitors thin.
                        Theoretical prediction (prob_pred) as dashed line if available.
        2 (top-right): log p(S | α) vs α — should be linear at small α.
        3 (bottom-left): Selectivity ratio p(S_0) / Σ_{S'≠S_0} p(S') vs α.
        4 (bottom-right): Effective number of states exp(H) vs α,
                           where H = -Σ p(S) log p(S).
                           Metastable boundary lines overlaid if provided.

    Args:
        leakage_df: Output of compute_leakage_profile.
        target: The targeted attractor.
        delta_star: Not used in plotting directly; kept for signature compatibility.
        metastable_boundaries: Output of find_metastable_boundary (dict of
            criterion -> alpha_boundary). Plotted as vertical lines in panel 4.
        output_path: If given, save figure to this path.
        top_n_competitors: Number of top-probability competitors to show.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()

    alpha_values = sorted(leakage_df["alpha"].unique())

    # Identify top competitors by mean probability across sweep
    competitor_df = leakage_df[leakage_df["attractor"] != target]
    mean_probs = (
        competitor_df.groupby("attractor")["prob"].mean().nlargest(top_n_competitors)
    )
    top_competitors = list(mean_probs.index)

    role_colors = {
        "target": "black",
        "bottleneck": "crimson",
        "partial": "steelblue",
        "orthogonal": "grey",
    }

    def get_series(attractor: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = leakage_df[leakage_df["attractor"] == attractor].sort_values("alpha")
        return (
            rows["alpha"].values,
            rows["prob"].values,
            rows["prob_pred"].values if "prob_pred" in rows else np.full(len(rows), np.nan),
        )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.flat

    # --- Panel 1: linear p ---
    alphas_t, probs_t, preds_t = get_series(target)
    ax1.plot(alphas_t, probs_t, color="black", lw=2.5, label=f"target {target}")
    if not np.all(np.isnan(preds_t)):
        ax1.plot(alphas_t, preds_t, color="black", lw=1.5, ls="--", alpha=0.6)
    for comp in top_competitors:
        role = leakage_df.loc[leakage_df["attractor"] == comp, "role"].iloc[0]
        color = role_colors.get(role, "grey")
        alphas_c, probs_c, preds_c = get_series(comp)
        ax1.plot(alphas_c, probs_c, color=color, lw=1.0, alpha=0.7, label=str(comp))
        if not np.all(np.isnan(preds_c)):
            ax1.plot(alphas_c, preds_c, color=color, lw=0.8, ls="--", alpha=0.4)
    ax1.set_xlabel("α (perturbation scale)")
    ax1.set_ylabel("p(S | α)")
    ax1.set_title("Probability vs. scale")
    ax1.legend(fontsize=7, loc="upper left")

    # --- Panel 2: log p ---
    ax2.plot(alphas_t, np.log(np.clip(probs_t, 1e-10, None)), color="black", lw=2.5)
    if not np.all(np.isnan(preds_t)):
        ax2.plot(alphas_t, np.log(np.clip(preds_t, 1e-10, None)), color="black", lw=1.5, ls="--", alpha=0.6)
    for comp in top_competitors:
        role = leakage_df.loc[leakage_df["attractor"] == comp, "role"].iloc[0]
        color = role_colors.get(role, "grey")
        alphas_c, probs_c, preds_c = get_series(comp)
        ax2.plot(alphas_c, np.log(np.clip(probs_c, 1e-10, None)), color=color, lw=1.0, alpha=0.7)
    ax2.set_xlabel("α (perturbation scale)")
    ax2.set_ylabel("log p(S | α)")
    ax2.set_title("Log-probability vs. scale  [linear model → straight lines]")

    # --- Panel 3: selectivity ratio ---
    selectivity = []
    for alpha in alpha_values:
        at_alpha = leakage_df[leakage_df["alpha"] == alpha]
        p_t = at_alpha.loc[at_alpha["attractor"] == target, "prob"].values
        p_others = at_alpha.loc[at_alpha["attractor"] != target, "prob"].sum()
        if len(p_t) > 0 and p_others > 0:
            selectivity.append(float(p_t[0]) / p_others)
        else:
            selectivity.append(np.nan)
    ax3.plot(alpha_values, selectivity, color="black", lw=2)
    ax3.axhline(1.0, color="grey", ls="--", lw=1, label="equal (p(S0) = Σ p(S'))")
    ax3.set_xlabel("α")
    ax3.set_ylabel("p(S_0) / Σ p(S')")
    ax3.set_title("Selectivity ratio")
    ax3.legend(fontsize=8)

    # --- Panel 4: effective number of states ---
    eff_n = []
    for alpha in alpha_values:
        at_alpha = leakage_df[leakage_df["alpha"] == alpha]
        p = at_alpha["prob"].values
        p = p[p > 0]
        if len(p) > 0:
            p = p / p.sum()
            H = -np.sum(p * np.log(p))
            eff_n.append(np.exp(H))
        else:
            eff_n.append(np.nan)
    ax4.plot(alpha_values, eff_n, color="steelblue", lw=2)
    ax4.set_xlabel("α")
    ax4.set_ylabel("exp(H)  [effective # states]")
    ax4.set_title("Effective number of states  [metastable: high; driven: low]")

    if metastable_boundaries:
        boundary_styles = {
            "target_dominates": ("black", "--", "target dominates"),
            "total_k_fraction": ("orangered", "-.", "k-hot fraction < 0.5"),
            "num_states": ("steelblue", ":", "n_states < threshold"),
        }
        for criterion, alpha_b in metastable_boundaries.items():
            if alpha_b is not None:
                style = boundary_styles.get(criterion, ("grey", "--", criterion))
                ax4.axvline(
                    alpha_b, color=style[0], ls=style[1], lw=1.5, label=style[2]
                )
        ax4.legend(fontsize=8)

    fig.suptitle(f"Leakage profile  |  target={target}", fontsize=11)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Sensitivity heatmap
# ---------------------------------------------------------------------------

def plot_sensitivity_heatmap(
    G: np.ndarray,
    attractors: list[tuple[int, ...]],
    C: int = 18,
    k_filter: int | None = 3,
    output_path: Path | str | None = None,
    max_attractors: int = 200,
) -> "plt.Figure":
    """Heatmap of G[S, c] with k-hot pattern overlaid.

    Rows: attractors sorted by size then lex order (truncated to max_attractors).
    Columns: clusters 0–C-1.
    Colourmap: diverging (RdBu_r), centred at 0.
    Overlay: white markers at positions where x_S = 1 (active clusters).

    Args:
        G: Sensitivity matrix of shape (N, C).
        attractors: Row labels, length N.
        C: Number of clusters.
        k_filter: If given, show only k-hot attractors.
        output_path: If given, save figure.
        max_attractors: Cap on number of rows displayed (highest |G| rows chosen).

    Returns:
        matplotlib Figure.
    """
    _require_mpl()

    # Optional k-filter
    if k_filter is not None:
        keep = [i for i, a in enumerate(attractors) if len(a) == k_filter]
        G = G[keep]
        attractors = [attractors[i] for i in keep]

    # Truncate for readability
    if len(attractors) > max_attractors:
        row_norms = np.linalg.norm(G, axis=1)
        top_idx = np.argsort(row_norms)[-max_attractors:][::-1]
        G = G[top_idx]
        attractors = [attractors[i] for i in top_idx]

    N = len(attractors)
    vmax = np.nanpercentile(np.abs(G), 95)

    fig, ax = plt.subplots(figsize=(max(8, C * 0.45), max(6, N * 0.06 + 1)))
    im = ax.imshow(
        G,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="∂ log p(S) / ∂ α_c")

    # Overlay k-hot pattern as white markers
    for row_i, att in enumerate(attractors):
        for c in att:
            ax.plot(c, row_i, "w.", markersize=3, alpha=0.7)

    ax.set_xlabel("Cluster index c")
    ax.set_ylabel(f"Attractor S (n={N})")
    ax.set_xticks(range(C))
    ax.set_title("Sensitivity matrix G  |  white dots = active clusters in x_S")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Capacity curve
# ---------------------------------------------------------------------------

def plot_capacity_curve(
    M_values: list[int],
    gamma_opt: list[float],
    C: int = 18,
    k: int = 3,
    gamma_random: list[float] | None = None,
    output_path: Path | str | None = None,
) -> "plt.Figure":
    """Plot Γ_min(W*; M) vs M with theoretical isotropic prediction.

    Isotropic prediction (permutation symmetry conjecture):
        γ_iso(M) = sqrt(2/k) · sqrt(M/C)

    This is the capacity curve under the assumption that the optimal subspace
    is isotropic (Π* = (M/C)·I), which is conjectured to hold for symmetric
    unperturbed networks.

    Args:
        M_values: List of M values.
        gamma_opt: Corresponding Γ_min(W*; M) values from the SDP.
        C: Number of clusters.
        k: Active clusters per attractor.
        gamma_random: Optional — mean Γ_min over random M-dim subspaces
            (from Monte Carlo), for comparison.
        output_path: If given, save figure.

    Returns:
        matplotlib Figure.
    """
    _require_mpl()

    M_arr = np.array(M_values, dtype=float)
    gamma_iso = np.sqrt(2 / k) * np.sqrt(M_arr / C)
    gamma_full = np.sqrt(2 / k)  # achieved at M = C

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(M_values, gamma_opt, "o-", color="black", lw=2, label="SDP optimal W*")
    ax.plot(M_arr, gamma_iso, "--", color="steelblue", lw=1.5,
            label=r"isotropic prediction $\sqrt{2/k}\cdot\sqrt{M/C}$")
    ax.axhline(gamma_full, color="grey", ls=":", lw=1,
               label=rf"full control ($M=C$): $\sqrt{{2/k}} = {gamma_full:.3f}$")

    if gamma_random is not None:
        ax.plot(M_values, gamma_random, "s--", color="orangered", lw=1.5,
                alpha=0.8, label="random subspace (mean)")

    ax.set_xlabel("M  (number of neuromodulatory modes)")
    ax.set_ylabel(r"$\Gamma_{\min}(W^*; M)$  (minimax targeting margin)")
    ax.set_title(f"Capacity curve  |  C={C}, k={k}")
    ax.set_xlim(0, max(M_values) + 0.5)
    ax.set_ylim(0, gamma_full * 1.15)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
