#!/usr/bin/env python3
"""Analyse the unified γ→π and single-mode limitations experiment.

Produces two figures:
  1. γ vs Δπ scatter — empirical mapping from geometric margin to occupancy change.
  2. coverage(M) curve — fraction of vocabulary targetable (γ > 0) vs M,
     for both optimized and random subspaces.

Reads:
  - SDP + random outputs from run_gamma_pi_experiment.py  (outputs/gamma_pi_setup/)
  - SNN validation results from run_capacity_validation.py (simulations/gamma_pi_validation/)
  - Baseline occupancy from unperturbed run                (simulations/baseline_run/)

Usage:
    python analyze_gamma_pi_experiment.py
    python analyze_gamma_pi_experiment.py \\
        --validation-dir simulations/gamma_pi_validation \\
        --setup-dir outputs/gamma_pi_setup \\
        --output-dir outputs/gamma_pi_analysis
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neuro_mod.core.spiking_net.analysis.snn_analyzer import SNNAnalyzer
from neuro_mod.core.spiking_net.analysis import helpers as _helpers

import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_setup(setup_dir: Path) -> dict:
    """Load all outputs from run_gamma_pi_experiment.py."""
    with open(setup_dir / "vocabulary.json") as f:
        vocabulary = [tuple(t) for t in json.load(f)]

    sdp = np.load(setup_dir / "sdp_outputs.npz")
    gammas_opt = np.load(setup_dir / "per_attractor_gammas_opt.npz")
    rand = np.load(setup_dir / "random_outputs.npz")

    return {
        "vocabulary": vocabulary,
        "sdp_M_values": sdp["M_values"].tolist(),
        "gamma_opt": sdp["gamma_opt"].tolist(),
        "Pi_matrices": sdp["Pi_matrices"],
        "per_attractor_gammas_opt": gammas_opt["gammas"],  # (n_M, n_vocab)
        "gammas_opt_M_values": gammas_opt["M_values"].tolist(),
        "random_M_values": rand["M_values"].tolist(),
        "random_seeds": rand["seeds"].tolist(),
        "random_gammas": rand["gammas"],             # (n_entries, n_vocab)
        "random_delta_stars": rand["delta_stars"],   # (n_entries, n_vocab, C)
    }


def load_baseline_pi(baseline_dir: Path) -> dict[tuple, float]:
    """Load baseline occupancy probabilities via SNNAnalyzer."""
    for subdir in [baseline_dir / "processed" / "sweep_0", baseline_dir / "processed"]:
        if (subdir / "attractors.npy").exists():
            analyzer = SNNAnalyzer(subdir)
            identities = _helpers.get_attractor_identities_in_order(
                analyzer.attractors_data
            )
            probs = analyzer.get_attractor_probs()
            return dict(zip(identities, probs))
    return {}


_CONFIG_RE = re.compile(r"^snn_validate_M(\d+)(?:_rand(\d+))?_(.+)\.yaml$")


def discover_validation_results(
    validation_dir: Path,
) -> list[dict]:
    """Walk validation results and return experiment records.

    Handles both optimized (M{M}/{label}/) and random (M{M}_rand{seed}/{label}/)
    directory layouts.

    Returns list of dicts with keys:
        M, label, target (tuple), path, direction_type ('optimized'|'random'),
        random_seed (int|None).
    """
    experiments = []
    if not validation_dir.exists():
        return experiments

    for m_dir in sorted(validation_dir.iterdir()):
        if not m_dir.is_dir():
            continue
        name = m_dir.name

        # Parse M and optional rand seed from directory name
        rand_match = re.match(r"^M(\d+)_rand(\d+)$", name)
        plain_match = re.match(r"^M(\d+)$", name)

        if rand_match:
            m_val = int(rand_match.group(1))
            rand_seed = int(rand_match.group(2))
            direction_type = "random"
        elif plain_match:
            m_val = int(plain_match.group(1))
            rand_seed = None
            direction_type = "optimized"
        else:
            continue

        for target_dir in sorted(m_dir.iterdir()):
            if not target_dir.is_dir():
                continue
            npy_path = target_dir / "processed" / "sweep_0" / "attractors.npy"
            if not npy_path.exists():
                continue
            label = target_dir.name
            try:
                target = tuple(int(x) for x in label.split("_"))
            except ValueError:
                continue

            experiments.append({
                "M": m_val,
                "target": target,
                "label": label,
                "path": target_dir,
                "direction_type": direction_type,
                "random_seed": rand_seed,
            })

    return experiments


def load_experiment_pi(exp: dict) -> dict[tuple, float]:
    """Load perturbed occupancy from a single experiment."""
    processed_dir = exp["path"] / "processed" / "sweep_0"
    analyzer = SNNAnalyzer(processed_dir)
    identities = _helpers.get_attractor_identities_in_order(analyzer.attractors_data)
    probs = analyzer.get_attractor_probs()
    return dict(zip(identities, probs))


# ---------------------------------------------------------------------------
# Figure 1: γ vs Δπ scatter
# ---------------------------------------------------------------------------

def plot_gamma_vs_delta_pi(
    setup: dict,
    experiments: list[dict],
    baseline_pi0: dict[tuple, float],
    output_dir: Path,
) -> None:
    """Scatter plot of per-attractor γ(S; M) vs Δπ(S) for each experiment.

    Colors by direction type (optimized vs random). Marker size by M.
    """
    vocabulary = setup["vocabulary"]
    vocab_index = {s: i for i, s in enumerate(vocabulary)}

    opt_gammas = setup["per_attractor_gammas_opt"]       # (n_M, n_vocab)
    opt_M_values = setup["gammas_opt_M_values"]
    opt_M_to_idx = {m: i for i, m in enumerate(opt_M_values)}

    rand_M_values = setup["random_M_values"]
    rand_seeds = setup["random_seeds"]
    rand_gammas = setup["random_gammas"]                 # (n_entries, n_vocab)

    # Collect points: (gamma, delta_pi, direction_type, M)
    points: list[dict] = []

    for exp in experiments:
        target = exp["target"]
        v_idx = vocab_index.get(target)
        if v_idx is None:
            continue

        M = exp["M"]
        dtype = exp["direction_type"]

        # Look up gamma
        if dtype == "optimized":
            m_idx = opt_M_to_idx.get(M)
            if m_idx is None:
                continue
            gamma = opt_gammas[m_idx, v_idx]
        else:
            # Find matching random entry
            seed = exp["random_seed"]
            gamma = None
            for r_idx, (rm, rs) in enumerate(zip(rand_M_values, rand_seeds)):
                if rm == M and rs == seed:
                    gamma = rand_gammas[r_idx, v_idx]
                    break
            if gamma is None:
                continue

        # Compute Δπ
        pi_perturbed = load_experiment_pi(exp)
        pi0 = baseline_pi0.get(target, 0.0)
        pi_pert = pi_perturbed.get(target, 0.0)
        delta_pi = pi_pert - pi0

        if np.isinf(gamma):
            continue

        points.append({
            "gamma": gamma,
            "delta_pi": delta_pi,
            "direction_type": dtype,
            "M": M,
        })

    if not points:
        print("[warn] No data points for γ vs Δπ scatter. Skipping.")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    for dtype, color, marker, label in [
        ("optimized", "black", "o", "SDP-optimized"),
        ("random", "tab:red", "x", "Random subspace"),
    ]:
        pts = [p for p in points if p["direction_type"] == dtype]
        if not pts:
            continue
        gs = np.array([p["gamma"] for p in pts])
        dps = np.array([p["delta_pi"] for p in pts])
        ms = np.array([p["M"] for p in pts])

        # Size proportional to M
        sizes = 20 + 8 * ms
        sc = ax.scatter(gs, dps, c=ms, s=sizes, marker=marker, alpha=0.6,
                        cmap="viridis", label=label, edgecolors="none")

    ax.set_xlabel(r"$\gamma(S;\, W)$ — targeting margin", fontsize=11)
    ax.set_ylabel(r"$\Delta\pi(S)$ — occupancy change", fontsize=11)
    ax.set_title(r"$\gamma \to \Delta\pi$ mapping", fontsize=12)
    ax.axhline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.3)
    ax.legend(fontsize=9)

    cbar = fig.colorbar(sc, ax=ax, label="M (modes)")
    fig.tight_layout()

    out_path = output_dir / "gamma_vs_delta_pi.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: coverage(M) curve
# ---------------------------------------------------------------------------

def plot_coverage_curve(
    setup: dict,
    output_dir: Path,
) -> None:
    """Plot fraction of vocabulary with γ > 0 as a function of M.

    Two curves: optimized (SDP) and random (mean ± std across seeds).
    """
    vocabulary = setup["vocabulary"]
    n_vocab = len(vocabulary)

    # Optimized
    opt_gammas = setup["per_attractor_gammas_opt"]  # (n_M, n_vocab)
    opt_M_values = np.array(setup["gammas_opt_M_values"])
    opt_coverage = np.sum(opt_gammas > 0, axis=1) / n_vocab
    opt_mean_gamma = np.array([
        np.mean(g[np.isfinite(g)]) if np.any(np.isfinite(g)) else 0.0
        for g in opt_gammas
    ])

    # Random: group by M, compute mean/std coverage across seeds
    rand_M_values = np.array(setup["random_M_values"])
    rand_gammas = setup["random_gammas"]  # (n_entries, n_vocab)
    unique_M = sorted(set(rand_M_values))

    rand_coverage_mean = []
    rand_coverage_std = []
    rand_gamma_mean = []
    rand_gamma_std = []
    for M in unique_M:
        mask = rand_M_values == M
        coverages = np.sum(rand_gammas[mask] > 0, axis=1) / n_vocab
        rand_coverage_mean.append(np.mean(coverages))
        rand_coverage_std.append(np.std(coverages))
        # Mean gamma (finite only)
        finite_gammas = rand_gammas[mask]
        means = []
        for row in finite_gammas:
            finite = row[np.isfinite(row)]
            means.append(np.mean(finite) if len(finite) > 0 else 0.0)
        rand_gamma_mean.append(np.mean(means))
        rand_gamma_std.append(np.std(means))

    rand_coverage_mean = np.array(rand_coverage_mean)
    rand_coverage_std = np.array(rand_coverage_std)
    rand_gamma_mean = np.array(rand_gamma_mean)
    rand_gamma_std = np.array(rand_gamma_std)
    unique_M = np.array(unique_M)

    # Plot: 2 panels — coverage and mean gamma
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Coverage
    ax1.plot(opt_M_values, opt_coverage, "o-", color="black", lw=2, ms=6,
             label="SDP-optimized", zorder=3)
    ax1.plot(unique_M, rand_coverage_mean, "s--", color="tab:red", lw=1.5, ms=5,
             label="Random (mean)", zorder=3)
    ax1.fill_between(unique_M,
                     rand_coverage_mean - rand_coverage_std,
                     rand_coverage_mean + rand_coverage_std,
                     color="tab:red", alpha=0.15, zorder=2)
    ax1.set_xlabel("M (neuromodulatory rank)", fontsize=11)
    ax1.set_ylabel("Coverage (fraction with γ > 0)", fontsize=11)
    ax1.set_title("Coverage vs. M", fontsize=12)
    ax1.set_ylim(-0.05, 1.1)
    ax1.axhline(1.0, color="k", lw=0.5, ls=":", alpha=0.3)
    ax1.legend(fontsize=9)

    # Panel 2: Mean precision (gamma)
    ax2.plot(opt_M_values, opt_mean_gamma, "o-", color="black", lw=2, ms=6,
             label="SDP-optimized", zorder=3)
    ax2.plot(unique_M, rand_gamma_mean, "s--", color="tab:red", lw=1.5, ms=5,
             label="Random (mean)", zorder=3)
    ax2.fill_between(unique_M,
                     rand_gamma_mean - rand_gamma_std,
                     rand_gamma_mean + rand_gamma_std,
                     color="tab:red", alpha=0.15, zorder=2)
    ax2.set_xlabel("M (neuromodulatory rank)", fontsize=11)
    ax2.set_ylabel(r"Mean $\gamma(S;\, W)$", fontsize=11)
    ax2.set_title("Precision vs. M", fontsize=12)
    ax2.legend(fontsize=9)

    fig.suptitle(
        "Single-mode limitations: coverage and precision vs. M\n"
        f"Vocabulary: {n_vocab} attractors (saturating set)",
        fontsize=11,
    )
    fig.tight_layout()

    out_path = output_dir / "coverage_precision_vs_M.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce γ→Δπ scatter and coverage(M) from gamma_pi experiment."
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default="simulations/gamma_pi_validation",
        help="Root directory with M{M}/{label}/ subdirectories.",
    )
    parser.add_argument(
        "--setup-dir",
        type=Path,
        default="outputs/gamma_pi_setup",
        help="Directory with outputs from run_gamma_pi_experiment.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default="outputs/gamma_pi_analysis",
        help="Where to save figures.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default="simulations/baseline_run",
        help="Unperturbed baseline run directory.",
    )
    args = parser.parse_args()

    def _resolve(p: Path) -> Path:
        return p if p.is_absolute() else (_ROOT / p)

    setup_dir = _resolve(args.setup_dir)
    validation_dir = _resolve(args.validation_dir)
    output_dir = _resolve(args.output_dir)
    baseline_dir = _resolve(args.baseline_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load setup data
    print(f"Loading setup from {setup_dir}")
    setup = load_setup(setup_dir)
    print(f"  Vocabulary: {len(setup['vocabulary'])} attractors")
    print(f"  SDP M range: {setup['sdp_M_values']}")
    print(f"  Random entries: {len(setup['random_M_values'])}")

    # Figure 2 (coverage) can be produced without SNN results
    print("\n--- Coverage & precision vs. M (from SDP/random, no SNN needed) ---")
    plot_coverage_curve(setup, output_dir)

    # Load baseline for Δπ
    print(f"\nLoading baseline from {baseline_dir}")
    baseline_pi0 = load_baseline_pi(baseline_dir)
    if baseline_pi0:
        print(f"  Baseline: {len(baseline_pi0)} attractors")
    else:
        print("  [warn] No baseline found. Δπ will use π₀=0.")

    # Discover and load validation results
    print(f"\nDiscovering experiments in {validation_dir}")
    experiments = discover_validation_results(validation_dir)
    if experiments:
        n_opt = sum(1 for e in experiments if e["direction_type"] == "optimized")
        n_rand = sum(1 for e in experiments if e["direction_type"] == "random")
        print(f"  Found: {n_opt} optimized, {n_rand} random")

        print("\n--- γ vs Δπ scatter ---")
        plot_gamma_vs_delta_pi(setup, experiments, baseline_pi0, output_dir)
    else:
        print("  No SNN results found. Run the simulations first.")
        print("  (Coverage plot was still generated from SDP data alone.)")


if __name__ == "__main__":
    main()
