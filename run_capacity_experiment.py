#!/usr/bin/env python3
"""Setup script for the capacity validation experiment.

Generates:
  1. A theoretical attractor vocabulary (30 k-hot attractors).
  2. SDP optimal projections Π*(M) for M ∈ {4, …, 12}.
  3. Targeting directions δ*_{S0,M} for every (M, S0) pair.
  4. One validation YAML config per (M, S0) pair.
  5. A self-contained output bundle (npz + json).

Usage:
    python run_capacity_experiment.py                     # defaults
    python run_capacity_experiment.py --n-vocab 12        # minimal saturating set only
    python run_capacity_experiment.py --out-dir outputs/capacity_setup
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. Vocabulary
# ---------------------------------------------------------------------------

def build_vocabulary(n: int = 30, C: int = 18, k: int = 3) -> list[tuple[int, ...]]:
    """Return n distinct k-hot attractor tuples (theoretical construction).

    Strategy:
      - k cyclic-shift partitions of [0..C-1] into k-groups → k × (C // k) attractors.
      - Equidistant step-C//k patterns → C // k more.
      - Fill remaining slots from lexicographic combinations.

    For C=18, k=3: 18 + 6 = 24 structured, then 6 from combinations → 30.
    """
    vocab: set[tuple[int, ...]] = set()
    clusters = list(range(C))

    for shift in range(k):
        for start in range(0, C, k):
            group = tuple(sorted((clusters[(start + shift + i) % C] for i in range(k))))
            if len(set(group)) == k:
                vocab.add(group)

    step = C // k
    for offset in range(step):
        group = tuple(sorted(((offset + i * step) % C) for i in range(k)))
        if len(set(group)) == k:
            vocab.add(group)

    for combo in combinations(range(C), k):
        if len(vocab) >= n:
            break
        vocab.add(combo)

    return sorted(vocab)[:n]


# ---------------------------------------------------------------------------
# 2. SDP
# ---------------------------------------------------------------------------

def run_sdp(
    vocabulary: list[tuple[int, ...]],
    M_range: range,
    C: int = 18,
    k: int = 3,
) -> dict:
    """Run capacity_curve SDP for each M and return the result dict."""
    from neuro_mod.analysis.capacity.sdp import capacity_curve

    print(f"Running SDP: {len(vocabulary)} attractors, M ∈ {list(M_range)} …")
    results = capacity_curve(vocabulary, M_range=M_range, C=C, k=k)

    for M, gamma, iso in zip(results["M_values"], results["gamma_opt"], results["is_isotropic"]):
        print(f"  M={M:2d}  γ={gamma:.4f}  isotropic={iso}")

    return results


# ---------------------------------------------------------------------------
# 3. Targeting directions
# ---------------------------------------------------------------------------

def compute_delta_stars(
    sdp_results: dict,
    vocabulary: list[tuple[int, ...]],
    C: int = 18,
) -> np.ndarray:
    """Compute δ*_{S0,M} for every (M, S0) pair.

    Returns:
        delta_stars: ndarray of shape (n_M, n_vocab, C).
            delta_stars[m_idx, v_idx] is the unit-norm targeting direction
            for vocabulary[v_idx] under M = M_values[m_idx].
            NaN when the attractor lies outside the subspace.
    """
    from neuro_mod.analysis.capacity.sdp import build_attractor_vectors, compute_targeting_direction

    X = build_attractor_vectors(vocabulary, C=C)
    n_M = len(sdp_results["M_values"])
    n_vocab = len(vocabulary)
    delta_stars = np.full((n_M, n_vocab, C), np.nan)

    for m_idx, (M, Pi) in enumerate(zip(sdp_results["M_values"], sdp_results["Pi_matrices"])):
        for v_idx, x_S0 in enumerate(X):
            try:
                delta_stars[m_idx, v_idx] = compute_targeting_direction(Pi, x_S0)
            except ValueError:
                pass  # NaN already set

    return delta_stars


# ---------------------------------------------------------------------------
# 4. Config generation
# ---------------------------------------------------------------------------

def generate_configs(
    sdp_results: dict,
    vocabulary: list[tuple[int, ...]],
    delta_stars: np.ndarray,
    base_config: Path,
    config_dir: Path,
    C: int = 18,
) -> list[Path]:
    """Write one validation YAML per (M, S0) pair.

    Returns list of written config paths.
    """
    from neuro_mod.analysis.capacity.generate_validation_configs import write_validation_yaml

    config_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for m_idx, M in enumerate(sdp_results["M_values"]):
        for v_idx, S0 in enumerate(vocabulary):
            delta = delta_stars[m_idx, v_idx]
            if np.any(np.isnan(delta)):
                continue

            label = "_".join(str(c) for c in S0)
            out_path = config_dir / f"snn_validate_M{M}_{label}.yaml"
            write_validation_yaml(
                base_config_path=base_config,
                output_path=out_path,
                delta_star=delta,
                label=label,
                M=M,
                C=C,
            )
            written.append(out_path)

    print(f"Wrote {len(written)} configs → {config_dir}")
    return written


# ---------------------------------------------------------------------------
# 5. Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    out_dir: Path,
    vocabulary: list[tuple[int, ...]],
    sdp_results: dict,
    delta_stars: np.ndarray,
) -> None:
    """Save vocabulary and SDP results to disk.

    Files written:
        vocabulary.json          — list of attractor tuples
        sdp_outputs.npz          — M_values, gamma_opt, Pi_matrices,
                                   is_isotropic, delta_stars, vocabulary_array
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # vocabulary.json — human-readable
    vocab_path = out_dir / "vocabulary.json"
    with open(vocab_path, "w") as f:
        json.dump([list(s) for s in vocabulary], f, indent=2)
    print(f"Saved vocabulary ({len(vocabulary)} attractors) → {vocab_path}")

    # sdp_outputs.npz — numerical bundle
    npz_path = out_dir / "sdp_outputs.npz"
    vocab_array = np.array([list(s) for s in vocabulary], dtype=np.int32)
    np.savez(
        npz_path,
        M_values=np.array(sdp_results["M_values"], dtype=np.int32),
        gamma_opt=np.array(sdp_results["gamma_opt"], dtype=np.float64),
        Pi_matrices=np.array(sdp_results["Pi_matrices"], dtype=np.float64),
        is_isotropic=np.array(sdp_results["is_isotropic"], dtype=bool),
        delta_stars=delta_stars,
        vocabulary=vocab_array,
    )
    print(f"Saved SDP outputs → {npz_path}")
    print(f"  Shapes: Pi_matrices={np.array(sdp_results['Pi_matrices']).shape}, "
          f"delta_stars={delta_stars.shape}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capacity experiment setup.")
    parser.add_argument(
        "--base-config",
        default="configs/snn_long_run_perturbed.yaml",
        help="Template YAML config.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/capacity_experiment",
        help="Directory for generated validation YAML configs.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/capacity_setup",
        help="Directory for SDP outputs (npz + json).",
    )
    parser.add_argument("--n-vocab", type=int, default=30, help="Vocabulary size.")
    parser.add_argument("--C", type=int, default=18, help="Number of clusters.")
    parser.add_argument("--k", type=int, default=3, help="Active clusters per attractor.")
    parser.add_argument("--M-min", type=int, default=4, help="Minimum M (modes).")
    parser.add_argument("--M-max", type=int, default=12, help="Maximum M (modes, inclusive).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    M_range = range(args.M_min, args.M_max + 1)

    base_config = ROOT / args.base_config
    config_dir = ROOT / args.config_dir
    out_dir = ROOT / args.out_dir

    vocabulary = build_vocabulary(n=args.n_vocab, C=args.C, k=args.k)
    print(f"Vocabulary: {len(vocabulary)} attractors")

    sdp_results = run_sdp(vocabulary, M_range=M_range, C=args.C, k=args.k)

    delta_stars = compute_delta_stars(sdp_results, vocabulary, C=args.C)

    generate_configs(
        sdp_results=sdp_results,
        vocabulary=vocabulary,
        delta_stars=delta_stars,
        base_config=base_config,
        config_dir=config_dir,
        C=args.C,
    )

    save_outputs(
        out_dir=out_dir,
        vocabulary=vocabulary,
        sdp_results=sdp_results,
        delta_stars=delta_stars,
    )


if __name__ == "__main__":
    main()
