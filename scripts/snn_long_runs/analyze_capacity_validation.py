#!/usr/bin/env python
"""Analyse capacity validation simulation data.

Walks simulations/capacity_validation/M{M}/{target_label}/ and runs:

  1. Landscape analysis (pooled across all experiments for a given M):
       - vocabulary selection, saturation coverage, SDP capacity curve
       - saves capacity_curve_M{M}.png, vocabulary_M{M}.json, coverage_M{M}.json

  2. Targeting analysis (per experiment):
       - attractor role classification relative to the target and δ*
       - saves {M}_{target_label}_targeting.csv

Usage::

    python scripts/snn_long_runs/analyze_capacity_validation.py
    python scripts/snn_long_runs/analyze_capacity_validation.py \\
        --validation-dir simulations/capacity_validation \\
        --output-dir simulations/capacity_validation/analysis \\
        --M 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Allow running from repo root without package install
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from neuro_mod.analysis.capacity import (
    build_attractor_vectors,
    capacity_curve,
    check_saturation_coverage,
    classify_attractor_role,
    load_attractors_from_npy,
    select_vocabulary_from_empirical,
)
import matplotlib.pyplot as plt
from scipy import stats

from neuro_mod.analysis.capacity.plotting import plot_capacity_curve


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
                target_tuple = tuple(int(x) for x in label.split("_"))
            except ValueError:
                print(f"  [warn] cannot parse target tuple from '{label}', skipping")
                continue
            experiments.append({
                "M": m_val,
                "target": target_tuple,
                "label": label,
                "path": target_dir,
            })

    return experiments


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_experiment(exp_info: dict, C: int = 18) -> dict:
    """Load attractor data and perturbation direction for one experiment.

    Reads:
      - processed/sweep_0/attractors.npy  via load_attractors_from_npy
      - metadata/sweep_0/config.yaml      for the δ* vector

    Returns exp_info extended with:
      attractor_tuples, probabilities, counts, meta, delta_star
    """
    tuples, probs, meta = load_attractors_from_npy(exp_info["path"])

    # Raw counts for pooling
    npy_path = exp_info["path"] / "processed" / "sweep_0" / "attractors.npy"
    att_dict = np.load(npy_path, allow_pickle=True).item()
    counts = {tup: int(info["#"]) for tup, info in att_dict.items()}
    count_arr = np.array([counts.get(t, 0) for t in tuples], dtype=np.int64)

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
        "attractor_tuples": tuples,
        "probabilities": probs,
        "counts": count_arr,
        "meta": meta,
        "delta_star": delta_star,
    }


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def pool_landscape(
    experiments: list[dict],
) -> tuple[list[tuple[int, ...]], np.ndarray]:
    """Pool attractor landscapes across experiments.

    Counts are summed across experiments for shared attractors; probability is
    re-estimated from pooled counts using a shared mean lifespan and total
    pooled duration.

    Returns:
        pooled_tuples:  sorted list of attractor tuples
        pooled_probs:   ndarray of p(S) estimates, re-normalised to sum ≤ 1
    """
    combined_counts: dict[tuple, int] = {}
    total_duration_ms = 0.0
    total_count = 0

    for exp in experiments:
        npy_path = exp["path"] / "processed" / "sweep_0" / "attractors.npy"
        att_dict = np.load(npy_path, allow_pickle=True).item()
        for tup, info in att_dict.items():
            combined_counts[tup] = combined_counts.get(tup, 0) + int(info["#"])
        total_duration_ms += exp["meta"]["total_duration_ms"]
        total_count += exp["meta"]["total_occurrences"]

    mean_lifespan_ms = total_duration_ms / total_count if total_count > 0 else 1.0

    pooled_tuples = sorted(combined_counts.keys())
    pooled_probs = np.array(
        [combined_counts[t] * mean_lifespan_ms / total_duration_ms for t in pooled_tuples],
        dtype=np.float64,
    )

    return pooled_tuples, pooled_probs


# ---------------------------------------------------------------------------
# Landscape analysis
# ---------------------------------------------------------------------------

def run_landscape_analysis(
    pooled_tuples: list[tuple[int, ...]],
    pooled_probs: np.ndarray,
    M_val: int,
    output_dir: Path,
    C: int = 18,
    k: int = 3,
) -> dict:
    """Select vocabulary, check coverage, compute SDP capacity curve, save outputs."""
    print(f"\n=== Landscape analysis (M={M_val}) ===")
    print(f"  Pooled attractor universe: {len(pooled_tuples)} attractors")

    vocab = select_vocabulary_from_empirical(
        pooled_tuples, pooled_probs, C=C, k=k, n_targets=12,
        strategy="greedy_saturating",
    )
    print(f"  Selected vocabulary: {len(vocab)} attractors")

    coverage = check_saturation_coverage(vocab, C=C)
    print(f"  Coverage: {coverage['n_covered']}/{coverage['n_total']} pairs "
          f"({'saturating' if coverage['is_saturating'] else 'NOT saturating'})")

    print(f"  Running SDP capacity curve for M=1..{C} ...")
    curve = capacity_curve(vocab, M_range=range(1, C + 1), C=C, k=k)

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.close("all")  # capacity curve figure saved later in unified plot

    vocab_path = output_dir / f"vocabulary_M{M_val}.json"
    with open(vocab_path, "w") as f:
        json.dump([list(t) for t in vocab], f, indent=2)
    print(f"  Saved: {vocab_path}")

    cov_path = output_dir / f"coverage_M{M_val}.json"
    cov_out = {k2: (v if not isinstance(v, list) else [list(x) for x in v])
               for k2, v in coverage.items()}
    with open(cov_path, "w") as f:
        json.dump(cov_out, f, indent=2)
    print(f"  Saved: {cov_path}")

    return {"vocab": vocab, "curve": curve, "coverage": coverage}


# ---------------------------------------------------------------------------
# Targeting analysis
# ---------------------------------------------------------------------------

def run_targeting_analysis(
    exp: dict,
    pooled_tuples: list[tuple[int, ...]],
    pooled_probs: np.ndarray,
    C: int = 18,
) -> pd.DataFrame:
    """Classify every pooled attractor's role relative to this experiment's target.

    Returns DataFrame with columns:
        attractor, prob, role, linear_score, score_contrast, direction, overlap, symmetric_diff
    sorted by prob descending.
    """
    target = exp["target"]
    delta_star = exp["delta_star"]

    target_x = build_attractor_vectors([target], C=C)[0]
    prob_map = dict(zip(pooled_tuples, pooled_probs))

    rows = []
    for tup in pooled_tuples:
        role_info = classify_attractor_role(tup, target, delta_star, target_x)
        rows.append({
            "attractor": str(tup),
            "prob": prob_map.get(tup, 0.0),
            "role": role_info["role"],
            "linear_score": role_info["linear_score"],
            "score_contrast": role_info["score_contrast"],
            "direction": role_info["direction"],
            "overlap": role_info["overlap"],
            "symmetric_diff": role_info["symmetric_diff"],
        })

    df = pd.DataFrame(rows).sort_values("prob", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _target_rank(df: pd.DataFrame, target: tuple) -> int:
    """Return 1-based rank of target attractor by probability."""
    target_str = str(target)
    matches = df[df["attractor"] == target_str]
    if matches.empty:
        return -1
    return int(matches.index[0]) + 1


def print_report(
    experiments: list[dict],
    landscape_results: dict[int, dict],
    targeting_results: list[tuple[dict, pd.DataFrame]],
) -> None:
    print("\n" + "=" * 60)
    print("CAPACITY VALIDATION ANALYSIS REPORT")
    print("=" * 60)

    for M_val, lr in sorted(landscape_results.items()):
        curve = lr["curve"]
        gamma_m1 = curve["gamma_opt"][0] if curve["gamma_opt"] else float("nan")
        gamma_full = (2 / 3) ** 0.5
        iso_m1 = gamma_full * (1 / 18) ** 0.5
        print(f"\nM={M_val}  |  Γ(M=1)={gamma_m1:.4f}  "
              f"(isotropic prediction: {iso_m1:.4f},  full: {gamma_full:.4f})")
        cov = lr["coverage"]
        print(f"  Vocabulary: {len(lr['vocab'])} attractors  |  "
              f"Coverage: {cov['n_covered']}/{cov['n_total']} "
              f"({'✓' if cov['is_saturating'] else '✗ NOT saturating'})")

    print("\nPer-experiment targeting summary:")
    for exp, df in targeting_results:
        target = exp["target"]
        rank = _target_rank(df, target)
        n_bottleneck = (df["role"] == "bottleneck").sum()
        target_prob = df[df["attractor"] == str(target)]["prob"].values
        tp_str = f"{target_prob[0]:.5f}" if len(target_prob) else "n/a"
        top_bn = df[df["role"] == "bottleneck"].head(3)["attractor"].tolist()
        print(f"  M={exp['M']}  target={target}  "
              f"rank={rank}  p(target)={tp_str}  "
              f"bottlenecks={n_bottleneck}  top3={top_bn}")

    print()


# ---------------------------------------------------------------------------
# Targeting quality plot
# ---------------------------------------------------------------------------

def plot_targeting_quality(
    targeting_results: list[tuple[dict, pd.DataFrame]],
    output_dir: Path,
) -> None:
    """Unified targeting quality summary with M on the x-axis.

    Two panels (stacked):
        Top:    target score percentile (%) — where the target sits in the
                linear-score distribution across all attractors (100% = highest)
        Bottom: Spearman ρ between linear_score and log p(S)

    One point per experiment, jittered horizontally.  Mean ± std shown per M.
    """
    rng = np.random.default_rng(0)

    # Collect per-experiment summary statistics
    records = []
    for exp, df in targeting_results:
        df_plot = df[df["prob"] > 0].copy()
        df_plot["log_prob"] = np.log(df_plot["prob"])
        target_str = str(exp["target"])
        is_target = df_plot["attractor"] == target_str

        rho, _ = stats.spearmanr(df_plot["linear_score"], df_plot["log_prob"])
        target_score = (df_plot.loc[is_target, "linear_score"].values[0]
                        if is_target.any() else np.nan)
        pct = ((df_plot["linear_score"] < target_score).mean() * 100
               if is_target.any() else np.nan)

        records.append({"M": exp["M"], "label": exp["label"], "rho": rho, "pct": pct})

    df_sum = pd.DataFrame(records)
    m_vals = sorted(df_sum["M"].unique())
    x_ticks = {mv: i for i, mv in enumerate(m_vals)}

    fig, (ax_pct, ax_rho) = plt.subplots(2, 1, figsize=(max(5, 2 * len(m_vals) + 2), 7),
                                          sharex=True)

    for ax, col, ylabel, title, ref in [
        (ax_pct, "pct",  "target score percentile (%)", "Target score percentile", 100),
        (ax_rho, "rho",  r"Spearman $\rho$",             r"Spearman $\rho$  (linear score vs log $p$)", 0),
    ]:
        ax.axhline(ref, color="k", lw=0.8, ls="--", alpha=0.4)

        for mv in m_vals:
            sub = df_sum[df_sum["M"] == mv][col].dropna().values
            if not len(sub):
                continue
            x0 = x_ticks[mv]
            jitter = rng.uniform(-0.15, 0.15, size=len(sub))
            ax.scatter(x0 + jitter, sub, s=40, color="steelblue",
                       alpha=0.7, linewidths=0, zorder=3)
            # Mean ± std marker
            ax.errorbar(x0, sub.mean(), yerr=sub.std(),
                        fmt="o", color="black", markersize=6,
                        capsize=4, lw=1.5, zorder=4)

        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11)

    ax_rho.set_xticks(list(x_ticks.values()))
    ax_rho.set_xticklabels([f"M={mv}" for mv in m_vals], fontsize=10)
    ax_rho.set_xlabel("M  (neuromodulatory rank)", fontsize=10)

    fig.suptitle("Targeting quality vs. M  |  α=0.025\n"
                 "Points = individual experiments.  Black = mean ± std.",
                 fontsize=11)
    fig.tight_layout()

    out_path = output_dir / "targeting_quality.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Δπ visualisation
# ---------------------------------------------------------------------------

_EXP_COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]


def plot_delta_pi(
    experiments: list[dict],
    baseline_pi0: dict,
    output_dir: Path,
    C: int = 18,
) -> None:
    """Bubble plot of Δπ = π(perturbed) − π₀(baseline), unified across all M values.

    Two panels (stacked):
        Top:    absolute Δπ  — mean per (M, group)
        Bottom: log(π/π₀)   — mean per (M, group), controls for baseline occupancy

    Groups: target (black), bottleneck/sym_diff=2 (crimson), other (steelblue).
    Bubble area ∝ std within group.  x-axis spans all M values found.
    """
    group_data: dict[str, list[dict]] = {"target": [], "bottleneck": [], "other": []}

    rng = np.random.default_rng(42)

    for exp_idx, exp in enumerate(experiments):
        target = exp["target"]
        delta_star = exp["delta_star"]
        target_x = build_attractor_vectors([target], C=C)[0]

        # Build π dict for this perturbed run
        npy_path = exp["path"] / "processed" / "sweep_0" / "attractors.npy"
        att_dict = np.load(npy_path, allow_pickle=True).item()
        total_dur = exp["meta"]["total_duration_ms"]
        total_occ = exp["meta"]["total_occurrences"]
        mean_ls = total_dur / total_occ if total_occ > 0 else 1.0

        pi_perturbed: dict[tuple, float] = {
            tup: info["#"] * mean_ls / total_dur
            for tup, info in att_dict.items()
        }

        # Union of attractors across baseline and perturbed
        all_atts = set(baseline_pi0.keys()) | set(pi_perturbed.keys())

        color = _EXP_COLORS[exp_idx % len(_EXP_COLORS)]

        for att in all_atts:
            pi0_val = baseline_pi0.get(att, 0.0)
            pi_val = pi_perturbed.get(att, 0.0)
            delta_pi_val = pi_val - pi0_val
            # log ratio only defined when both are positive
            log_ratio = (np.log(pi_val / pi0_val)
                         if pi0_val > 0 and pi_val > 0 else np.nan)

            role_info = classify_attractor_role(att, target, delta_star, target_x)
            sym_diff = role_info["symmetric_diff"]

            if att == target:
                group = "target"
            elif sym_diff == 2:
                group = "bottleneck"
            else:
                group = "other"

            group_data[group].append({
                "M": exp["M"],
                "delta_pi": delta_pi_val,
                "log_ratio": log_ratio,
                "pi0": pi0_val,
                "exp_label": exp["label"],
                "color": color,
                "exp_idx": exp_idx,
            })

    # ---------- summarise: mean ± std per (M, group) ----------
    group_keys = ["target", "bottleneck", "other"]
    group_colors = {"target": "black", "bottleneck": "crimson", "other": "steelblue"}
    group_labels = {"target": "Target", "bottleneck": "Bottleneck (sym_diff=2)", "other": "Other"}
    # Slight x-offsets so the three groups don't overlap at the same M tick
    group_offsets = {"target": -0.15, "bottleneck": 0.0, "other": 0.15}

    m_vals_all = sorted({e["M"] for e in experiments})
    x_ticks = {mv: i for i, mv in enumerate(m_vals_all)}

    # For size scaling: collect all stds first, then normalise
    summary: dict[str, dict] = {}   # key: group_key → {M → (mean_abs, std_abs, mean_rel, std_rel)}
    for group_key in group_keys:
        summary[group_key] = {}
        for mv in m_vals_all:
            rows = [r for r in group_data[group_key] if r["M"] == mv]
            if not rows:
                continue
            abs_vals = np.array([r["delta_pi"] for r in rows])
            rel_vals = np.array([r["log_ratio"] for r in rows], dtype=float)
            rel_vals = rel_vals[np.isfinite(rel_vals)]
            summary[group_key][mv] = {
                "mean_abs": float(np.mean(abs_vals)),
                "std_abs": float(np.std(abs_vals)),
                "mean_rel": float(np.mean(rel_vals)) if len(rel_vals) else np.nan,
                "std_rel": float(np.std(rel_vals)) if len(rel_vals) else np.nan,
            }

    # Global max std for each metric → used to normalise bubble sizes
    all_std_abs = [s["std_abs"] for g in summary.values() for s in g.values()]
    all_std_rel = [s["std_rel"] for g in summary.values() for s in g.values()
                   if np.isfinite(s["std_rel"])]
    max_std_abs = max(all_std_abs) if all_std_abs else 1.0
    max_std_rel = max(all_std_rel) if all_std_rel else 1.0
    MIN_S, MAX_S = 60, 800   # marker area range

    def _size(std, max_std):
        return MIN_S + (std / max_std) * (MAX_S - MIN_S)

    # ---------- plot: two panels stacked ----------
    fig, (ax_abs, ax_rel) = plt.subplots(2, 1, figsize=(6, 8))

    for ax in (ax_abs, ax_rel):
        ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4, zorder=1)

    legend_handles = []
    for group_key in group_keys:
        color = group_colors[group_key]
        for mv, s in summary[group_key].items():
            x = x_ticks[mv] + group_offsets[group_key]

            ax_abs.scatter(x, s["mean_abs"],
                           s=_size(s["std_abs"], max_std_abs),
                           c=color, alpha=0.8, linewidths=0.5,
                           edgecolors="white", zorder=3)

            if np.isfinite(s["mean_rel"]):
                ax_rel.scatter(x, s["mean_rel"],
                               s=_size(s["std_rel"], max_std_rel),
                               c=color, alpha=0.8, linewidths=0.5,
                               edgecolors="white", zorder=3)

        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=color, markersize=9,
                       label=group_labels[group_key])
        )

    for ax in (ax_abs, ax_rel):
        ax.set_xticks(list(x_ticks.values()))
        ax.set_xticklabels([f"M={mv}" for mv in m_vals_all], fontsize=10)
        ax.set_xlim(-0.6, len(m_vals_all) - 0.4)

    ax_abs.set_ylabel(r"mean $\Delta\pi$  (absolute)", fontsize=10)
    ax_rel.set_ylabel(r"mean $\log(\pi / \pi_0)$  (relative)", fontsize=10)
    ax_abs.set_title("Absolute occupancy change", fontsize=11)
    ax_rel.set_title("Relative occupancy change", fontsize=11)

    # Size legend (approximate)
    size_legend = [
        plt.scatter([], [], s=_size(f * max_std_abs, max_std_abs),
                    c="grey", alpha=0.6, label=f"std = {f:.0%} max")
        for f in [0.25, 0.5, 1.0]
    ]
    ax_abs.legend(handles=legend_handles + size_legend,
                  fontsize=8, loc="upper right",
                  title="group  /  bubble ∝ std", title_fontsize=8)

    fig.suptitle(
        r"$\Delta\pi$ per attractor role  |  α=0.025 vs. baseline" "\n"
        "Bubble centre = group mean.  Bubble area ∝ std within group.",
        fontsize=10,
    )
    fig.tight_layout()

    out_path = output_dir / "delta_pi.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Unified capacity curve
# ---------------------------------------------------------------------------

def plot_capacity_curve_unified(
    experiments: list[dict],
    landscape_results: dict[int, dict],
    output_dir: Path,
    C: int = 18,
    k: int = 3,
) -> None:
    """Single capacity curve figure, one line per M group (from its vocabulary).

    If all M groups produce the same curve (same landscape), lines will overlap —
    that itself is informative.  Each curve is labelled by which M group's vocabulary
    was used to compute it.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    # Reference lines (same for all)
    gamma_full = (2 / 3) ** 0.5
    m_range = list(range(1, C + 1))
    iso = [gamma_full * (m / C) ** 0.5 for m in m_range]
    ax.plot(m_range, iso, "b--", lw=1.2, alpha=0.6,
            label=r"isotropic $\sqrt{M/C}$")
    ax.axhline(gamma_full, color="b", lw=0.8, ls=":", alpha=0.4,
               label=f"full control = {gamma_full:.3f}")

    colors = plt.cm.tab10.colors
    for i, (M_val, lr) in enumerate(sorted(landscape_results.items())):
        curve = lr["curve"]
        ax.plot(curve["M_values"], curve["gamma_opt"],
                "o-", color=colors[i % len(colors)], lw=1.8, ms=5,
                label=f"SDP optimal (vocab from M={M_val} exps)")

    ax.set_xlabel("M  (number of neuromodulatory modes)", fontsize=11)
    ax.set_ylabel(r"$\Gamma_\mathrm{opt}(M)$  (minimax targeting margin)", fontsize=11)
    ax.set_title(f"Capacity curve  |  C={C}, k={k}", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_path = output_dir / "capacity_curve.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyse capacity validation simulation data."
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
        "--k",
        type=int,
        default=3,
        help="Active clusters per attractor (default: 3).",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=_ROOT / "simulations" / "baseline_run",
        help="Directory of the unperturbed baseline run (default: simulations/baseline_run).",
    )
    args = parser.parse_args()

    validation_dir = args.validation_dir.resolve()
    output_dir = (args.output_dir or validation_dir / "analysis").resolve()

    baseline_dir = args.baseline_dir.resolve()

    print(f"Validation dir : {validation_dir}")
    print(f"Output dir     : {output_dir}")
    print(f"Baseline dir   : {baseline_dir}")

    # Load baseline π₀ once.
    # The baseline run may have a flat processed/ layout (no sweep_0/ subdir).
    baseline_pi0: dict[tuple, float] = {}
    if baseline_dir.exists():
        # Try sweep_0 layout first, then flat processed/ layout
        bl_npy = baseline_dir / "processed" / "sweep_0" / "attractors.npy"
        bl_cfg = baseline_dir / "processed" / "sweep_0" / "processor_config.json"
        if not bl_npy.exists():
            bl_npy = baseline_dir / "processed" / "attractors.npy"
            bl_cfg = baseline_dir / "processed" / "processor_config.json"

        if bl_npy.exists() and bl_cfg.exists():
            bl_att = np.load(bl_npy, allow_pickle=True).item()
            import json as _json
            with open(bl_cfg) as _f:
                bl_meta = _json.load(_f)
            total_dur_bl = float(bl_meta["total_duration_ms"])
            total_occ_bl = sum(int(v["#"]) for v in bl_att.values())
            mean_ls_bl = total_dur_bl / total_occ_bl if total_occ_bl > 0 else 1.0
            baseline_pi0 = {
                tup: int(info["#"]) * mean_ls_bl / total_dur_bl
                for tup, info in bl_att.items()
            }
            print(f"Baseline: {len(baseline_pi0)} attractors loaded")
        else:
            print(f"[warn] Cannot find attractors.npy under {baseline_dir}. Δπ plot will use π₀=0.")
    else:
        print(f"[warn] Baseline dir not found: {baseline_dir}. Δπ plot will use π₀=0.")

    # Discover
    experiments_raw = discover_experiments(validation_dir, M_filter=args.M)
    if not experiments_raw:
        print("No experiments found. Check --validation-dir.")
        return
    print(f"Found {len(experiments_raw)} experiments: "
          + ", ".join(f"M{e['M']}/{e['label']}" for e in experiments_raw))

    # Load
    experiments = [load_experiment(e, C=args.C) for e in experiments_raw]

    # Group by M
    m_values = sorted({e["M"] for e in experiments})
    landscape_results: dict[int, dict] = {}
    targeting_results: list[tuple[dict, pd.DataFrame]] = []

    for M_val in m_values:
        exps_m = [e for e in experiments if e["M"] == M_val]

        # Landscape (pooled)
        pooled_tuples, pooled_probs = pool_landscape(exps_m)
        lr = run_landscape_analysis(
            pooled_tuples, pooled_probs, M_val, output_dir,
            C=args.C, k=args.k,
        )
        landscape_results[M_val] = lr

        # Targeting (per experiment)
        for exp in exps_m:
            print(f"\n--- Targeting: M={M_val}  target={exp['target']} ---")
            df = run_targeting_analysis(exp, pooled_tuples, pooled_probs, C=args.C)
            csv_path = output_dir / f"M{M_val}_{exp['label']}_targeting.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")
            targeting_results.append((exp, df))

    # Unified plots (all M values on one figure each)
    print("\n--- Generating unified plots ---")
    plot_capacity_curve_unified(experiments, landscape_results, output_dir,
                                C=args.C, k=args.k)
    plot_targeting_quality(targeting_results, output_dir)
    plot_delta_pi(experiments, baseline_pi0, output_dir, C=args.C)

    print_report(experiments, landscape_results, targeting_results)


if __name__ == "__main__":
    main()
