#!/usr/bin/env python
"""Analyse capacity validation simulation data.

Walks simulations/capacity_validation/M{M}/{target_label}/ and produces
a single figure — delta_pi_curve.png — showing empirical Δπ(M) vs. the
theoretical SDP capacity curve.

π(attractor) is computed as the exact occupancy probability:
    π(S) = total_time_in_S / total_simulation_duration

using SNNAnalyzer, which reads the per-attractor total_duration field
written by the SNN processor.

Usage::

    python scripts/snn_long_runs/analyze_capacity_validation.py
    python scripts/snn_long_runs/analyze_capacity_validation.py \\
        --validation-dir simulations/capacity_validation \\
        --output-dir simulations/capacity_validation/analysis \\
        --sdp-dir outputs/capacity_setup \\
        --M 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow running from repo root without package install
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neuro_mod.analysis.capacity import (
    build_attractor_vectors,
    classify_attractor_role,
)
from neuro_mod.core.spiking_net.analysis.snn_analyzer import SNNAnalyzer
from neuro_mod.core.spiking_net.analysis import helpers as _helpers
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# SDP outputs loader
# ---------------------------------------------------------------------------

def load_sdp_outputs(sdp_dir: Path) -> dict:
    """Load the canonical vocabulary and pre-computed SDP curve from run_capacity_experiment.py.

    Reads:
        vocabulary.json   — list of attractor tuples
        sdp_outputs.npz   — M_values, gamma_opt, Pi_matrices, is_isotropic

    Returns dict with keys:
        vocabulary  (list[tuple[int, ...]])
        curve       (dict with keys M_values, gamma_opt, is_isotropic, Pi_matrices)
    """
    vocab_path = sdp_dir / "vocabulary.json"
    npz_path = sdp_dir / "sdp_outputs.npz"

    if not vocab_path.exists() or not npz_path.exists():
        raise FileNotFoundError(
            f"SDP outputs not found in {sdp_dir}. "
            "Run run_capacity_experiment.py first."
        )

    with open(vocab_path) as f:
        vocabulary = [tuple(t) for t in json.load(f)]

    npz = np.load(npz_path)
    curve = {
        "M_values": npz["M_values"].tolist(),
        "gamma_opt": npz["gamma_opt"].tolist(),
        "is_isotropic": npz["is_isotropic"].tolist(),
        "Pi_matrices": npz["Pi_matrices"],
    }

    return {"vocabulary": vocabulary, "curve": curve}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_experiments(
    validation_dir: Path,
    M_filter: int | None = None,
) -> list[dict]:
    """Walk validation_dir/M{M}/{target_label}/ and return experiment records.

    Returns list of dicts with keys: M (int), target (tuple), label (str), path (Path).
    Only directories containing processed/sweep_0/attractors.npy are included.
    """
    experiments = []
    for m_dir in sorted(validation_dir.iterdir()):
        if not m_dir.is_dir() or not m_dir.name.startswith("M"):
            continue
        try:
            m_val = int(m_dir.name[1:])
        except ValueError:
            continue
        if M_filter is not None and m_val != M_filter:
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
            })

    return experiments


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_processed_dir(base_dir: Path) -> Path | None:
    """Return the processed data directory under base_dir, or None if not found.

    Tries processed/sweep_0/ first, then processed/ as fallback.
    """
    sweep0 = base_dir / "processed" / "sweep_0"
    if (sweep0 / "attractors.npy").exists():
        return sweep0
    flat = base_dir / "processed"
    if (flat / "attractors.npy").exists():
        return flat
    return None


def _analyzer_pi(processed_dir: Path) -> dict[tuple, float]:
    """Load attractor occupancy probabilities from processed_dir using SNNAnalyzer.

    Returns dict mapping attractor identity tuple → π(attractor), where
        π(S) = total_time_in_S_ms / total_simulation_duration_ms

    This is the exact time-fraction definition (not a count-based approximation).
    """
    analyzer = SNNAnalyzer(processed_dir)
    identities = _helpers.get_attractor_identities_in_order(analyzer.attractors_data)
    probs = analyzer.get_attractor_probs()
    return dict(zip(identities, probs))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_experiment(exp_info: dict, C: int = 18) -> dict:
    """Load attractor occupancy probabilities and perturbation direction for one experiment.

    Reads:
      - processed/sweep_0/        via SNNAnalyzer  (exact π = total_time / total_duration)
      - metadata/sweep_0/config.yaml               for the δ* vector

    Returns exp_info extended with:
      pi_perturbed  (dict[tuple, float])   exact occupancy probabilities
      delta_star    (ndarray, shape C)     perturbation direction
    """
    processed_dir = exp_info["path"] / "processed" / "sweep_0"
    pi_perturbed = _analyzer_pi(processed_dir)

    # δ* from config.yaml
    config_path = exp_info["path"] / "metadata" / "sweep_0" / "config.yaml"
    delta_star = np.zeros(C, dtype=np.float64)
    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        vectors = cfg.get("perturbation", {}).get("rate", {}).get("vectors")
        if vectors:
            delta_star = np.array(vectors[0], dtype=np.float64)

    return {
        **exp_info,
        "pi_perturbed": pi_perturbed,
        "delta_star": delta_star,
    }


# ---------------------------------------------------------------------------
# Δπ(M) curve — empirical analog of the capacity curve
# ---------------------------------------------------------------------------

def plot_delta_pi_curve(
    experiments: list[dict],
    baseline_pi0: dict,
    sdp_curve: dict,
    output_dir: Path,
    C: int = 18,
) -> None:
    """Line plot of mean Δπ and mean log(π/π₀) as a function of M.

    Two panels (stacked):
        Top:    mean Δπ  (absolute) ± std  per group, vs M
        Bottom: mean log(π/π₀) (relative) ± std per group, vs M
                with γ_opt(M) overlaid on a secondary y-axis

    Groups: target (black), bottleneck/sym_diff=2 (crimson), other (steelblue).
    Shaded bands = ±1 std across experiments at each M.

    sdp_curve: dict with keys M_values (list[int]) and gamma_opt (list[float]).
    """
    group_data: dict[str, list[dict]] = {"target": [], "bottleneck": [], "other": []}

    for exp in experiments:
        target = exp["target"]
        delta_star = exp["delta_star"]
        target_x = build_attractor_vectors([target], C=C)[0]
        pi_perturbed = exp["pi_perturbed"]

        all_atts = set(baseline_pi0.keys()) | set(pi_perturbed.keys())
        for att in all_atts:
            pi0_val = baseline_pi0.get(att, 0.0)
            pi_val = pi_perturbed.get(att, 0.0)
            delta_pi_val = pi_val - pi0_val
            log_ratio = (np.log(pi_val / pi0_val)
                         if pi0_val > 0 and pi_val > 0 else np.nan)
            role_info = classify_attractor_role(att, target, delta_star, target_x)
            if att == target:
                group = "target"
            elif role_info["symmetric_diff"] == 2:
                group = "bottleneck"
            else:
                group = "other"
            group_data[group].append({
                "M": exp["M"], "delta_pi": delta_pi_val, "log_ratio": log_ratio,
            })

    # ---- summarise: mean ± std per (M, group) --------------------------------
    group_keys = ["target", "bottleneck", "other"]
    group_colors = {"target": "black", "bottleneck": "crimson", "other": "steelblue"}
    group_labels = {"target": "Target", "bottleneck": "Bottleneck (sym_diff=2)",
                    "other": "Other"}
    m_vals = sorted({e["M"] for e in experiments})

    summary: dict[str, dict] = {}
    for gk in group_keys:
        summary[gk] = {}
        for mv in m_vals:
            rows = [r for r in group_data[gk] if r["M"] == mv]
            if not rows:
                continue
            abs_vals = np.array([r["delta_pi"] for r in rows])
            rel_vals = np.array([r["log_ratio"] for r in rows], dtype=float)
            rel_vals_finite = rel_vals[np.isfinite(rel_vals)]
            summary[gk][mv] = {
                "mean_abs": float(np.mean(abs_vals)),
                "std_abs": float(np.std(abs_vals)),
                "mean_rel": float(np.mean(rel_vals_finite)) if len(rel_vals_finite) else np.nan,
                "std_rel": float(np.std(rel_vals_finite)) if len(rel_vals_finite) else np.nan,
            }

    # ---- plot ----------------------------------------------------------------
    fig, (ax_abs, ax_rel) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    for ax in (ax_abs, ax_rel):
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.3, zorder=1)

    legend_handles = []
    for gk in group_keys:
        color = group_colors[gk]
        mv_arr = np.array([mv for mv in m_vals if mv in summary[gk]])
        if not len(mv_arr):
            continue

        mean_abs = np.array([summary[gk][mv]["mean_abs"] for mv in mv_arr])
        std_abs  = np.array([summary[gk][mv]["std_abs"]  for mv in mv_arr])
        mean_rel = np.array([summary[gk][mv]["mean_rel"] for mv in mv_arr])
        std_rel  = np.array([summary[gk][mv]["std_rel"]  for mv in mv_arr])

        for ax, mean, std in [(ax_abs, mean_abs, std_abs),
                               (ax_rel, mean_rel, std_rel)]:
            finite = np.isfinite(mean)
            if not finite.any():
                continue
            ax.plot(mv_arr[finite], mean[finite],
                    "o-", color=color, lw=2, ms=5, zorder=3)
            ax.fill_between(mv_arr[finite],
                            mean[finite] - std[finite],
                            mean[finite] + std[finite],
                            color=color, alpha=0.15, zorder=2)

        legend_handles.append(
            plt.Line2D([0], [0], color=color, lw=2, marker="o", ms=5,
                       label=group_labels[gk])
        )

    # secondary axis: γ_opt(M)
    ax2 = ax_rel.twinx()
    ax2.plot(sdp_curve["M_values"], sdp_curve["gamma_opt"],
             "s--", color="tab:orange", lw=1.5, ms=4, alpha=0.8,
             label=r"$\Gamma_\mathrm{opt}(M)$ (SDP)")
    ax2.set_ylabel(r"$\Gamma_\mathrm{opt}(M)$", fontsize=10, color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(bottom=0)
    legend_handles.append(
        plt.Line2D([0], [0], color="tab:orange", lw=1.5, ls="--", marker="s",
                   ms=4, label=r"$\Gamma_\mathrm{opt}$ (SDP, right axis)")
    )

    ax_abs.set_ylabel(r"mean $\Delta\pi$  (absolute)", fontsize=10)
    ax_rel.set_ylabel(r"mean $\log(\pi / \pi_0)$  (relative)", fontsize=10)
    ax_abs.set_title(r"$\Delta\pi(M)$ — absolute occupancy change vs. M", fontsize=11)
    ax_rel.set_title(r"$\Delta\pi(M)$ — relative occupancy change vs. M", fontsize=11)

    ax_rel.set_xlabel("M  (neuromodulatory rank)", fontsize=10)
    ax_rel.set_xticks(m_vals)

    ax_abs.legend(handles=legend_handles, fontsize=8, loc="upper left")

    fig.suptitle(
        r"Empirical $\Delta\pi(M)$  |  α=0.25 vs. baseline" "\n"
        r"Lines = group mean.  Bands = ±1 std across experiments.",
        fontsize=10,
    )
    fig.tight_layout()

    out_path = output_dir / "delta_pi_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce delta_pi_curve.png from capacity validation simulations."
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=_ROOT / "simulations" / "capacity_validation",
        help="Root directory containing M{M}/{target_label}/ subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to save results. Default: {validation-dir}/analysis/.",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=None,
        help="Restrict analysis to this M value (default: all).",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=18,
        help="Number of excitatory clusters (default: 18).",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=_ROOT / "simulations" / "baseline_run",
        help="Directory of the unperturbed baseline run (default: simulations/baseline_run).",
    )
    parser.add_argument(
        "--sdp-dir",
        type=Path,
        default=_ROOT / "outputs" / "capacity_setup",
        help="Directory containing vocabulary.json and sdp_outputs.npz from "
             "run_capacity_experiment.py (default: outputs/capacity_setup).",
    )
    args = parser.parse_args()

    def _resolve(p: Path) -> Path:
        """Resolve relative paths against _ROOT (like the defaults do)."""
        return p if p.is_absolute() else (_ROOT / p)

    validation_dir = _resolve(args.validation_dir)
    output_dir = _resolve(args.output_dir) if args.output_dir else validation_dir / "analysis"
    baseline_dir = _resolve(args.baseline_dir)
    sdp_dir = _resolve(args.sdp_dir)

    print(f"Validation dir : {validation_dir}")
    print(f"Output dir     : {output_dir}")
    print(f"Baseline dir   : {baseline_dir}")
    print(f"SDP dir        : {sdp_dir}")

    # Load canonical vocabulary and SDP curve.
    sdp_data = load_sdp_outputs(sdp_dir)
    curve = sdp_data["curve"]
    print(f"Vocabulary     : {len(sdp_data['vocabulary'])} attractors")
    print(f"SDP M range    : M={curve['M_values'][0]}..{curve['M_values'][-1]}")

    # Load baseline π₀ via SNNAnalyzer (exact: total_time / total_duration).
    baseline_pi0: dict[tuple, float] = {}
    if baseline_dir.exists():
        bl_processed = _find_processed_dir(baseline_dir)
        if bl_processed is not None:
            baseline_pi0 = _analyzer_pi(bl_processed)
            print(f"Baseline: {len(baseline_pi0)} attractors loaded")
        else:
            print(f"[warn] Cannot find attractors.npy under {baseline_dir}. Δπ plot will use π₀=0.")
    else:
        print(f"[warn] Baseline dir not found: {baseline_dir}. Δπ plot will use π₀=0.")

    # Discover experiments.
    if not validation_dir.exists():
        print(f"[error] Validation dir not found: {validation_dir}")
        return
    experiments_raw = discover_experiments(validation_dir, M_filter=args.M)
    if not experiments_raw:
        print("No experiments found. Check --validation-dir.")
        return
    print(f"Found {len(experiments_raw)} experiments: "
          + ", ".join(f"M{e['M']}/{e['label']}" for e in experiments_raw))

    # Load each experiment (exact π via SNNAnalyzer).
    experiments = [load_experiment(e, C=args.C) for e in experiments_raw]

    # Produce the single output figure.
    output_dir.mkdir(parents=True, exist_ok=True)
    print("\n--- Generating delta_pi_curve ---")
    plot_delta_pi_curve(experiments, baseline_pi0, curve, output_dir, C=args.C)


if __name__ == "__main__":
    main()
