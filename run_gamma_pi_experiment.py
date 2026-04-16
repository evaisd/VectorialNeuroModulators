#!/usr/bin/env python3
"""Setup script for the unified γ→π mapping and single-mode limitations experiment.

Generates:
  1. SDP-optimal projections and per-attractor γ(S; M) for M ∈ {1, …, M_max}.
  2. Random (non-optimized) projection directions for comparison.
  3. One validation YAML config per (M, S0, direction_type) triple.
  4. A self-contained output bundle with per-attractor margins.

The generated configs are run via the existing run_capacity_validation.py script.
Analysis is done by analyze_gamma_pi_experiment.py.

Informed by: raw/notes/capacity/Next Steps 16.04.md

Usage:
    python run_gamma_pi_experiment.py
    python run_gamma_pi_experiment.py --M-min 1 --M-max 12 --n-random 3
    python run_gamma_pi_experiment.py --alpha 0.25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from run_capacity_experiment import (
    build_saturating_vocabulary,
    build_vocabulary,
    run_sdp,
    save_outputs,
)
from neuro_mod.analysis.capacity.sdp import (
    build_attractor_vectors,
    build_difference_vectors,
    compute_targeting_direction,
)
from neuro_mod.analysis.capacity.generate_validation_configs import (
    write_validation_yaml,
)


# ---------------------------------------------------------------------------
# Per-attractor margin computation
# ---------------------------------------------------------------------------

def compute_per_attractor_gamma(
    Pi: np.ndarray,
    vocabulary: list[tuple[int, ...]],
    C: int = 18,
    k: int = 3,
) -> np.ndarray:
    """Compute γ(S; W) for each attractor S in vocabulary under projection Pi.

    γ(S; W) = min_{S' in H_k \\ {S}} ||Π_W d_{SS'}||_2

    For efficiency, only bottleneck pairs (|S △ S'| = 2) are checked since
    they are the binding constraints.

    Returns:
        gammas: ndarray of shape (n_vocab,). γ(S) per attractor.
            inf if no bottleneck competitor exists in the full H_k.
    """
    X = build_attractor_vectors(vocabulary, C=C)
    n_vocab = len(vocabulary)
    gammas = np.full(n_vocab, np.inf)

    # Build all bottleneck difference vectors from vocab against full H_k
    vocab_set = set(vocabulary)
    for v_idx, S in enumerate(vocabulary):
        s_set = set(S)
        min_margin = np.inf
        for i in s_set:
            for j in range(C):
                if j in s_set:
                    continue
                S_prime = tuple(sorted((s_set - {i}) | {j}))
                d = (X[v_idx] - _khot(S_prime, C)) / np.sqrt(k)
                proj_norm = np.sqrt(d @ Pi @ d)
                if proj_norm < min_margin:
                    min_margin = proj_norm
        gammas[v_idx] = min_margin

    return gammas


def _khot(S: tuple[int, ...], C: int) -> np.ndarray:
    """Build k-hot vector for attractor S."""
    x = np.zeros(C)
    for i in S:
        x[i] = 1.0
    return x


# ---------------------------------------------------------------------------
# Random direction generation
# ---------------------------------------------------------------------------

def generate_random_directions(
    vocabulary: list[tuple[int, ...]],
    M: int,
    C: int = 18,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate random targeting directions for each attractor.

    For each attractor S, draw a random M-dimensional subspace (uniform on
    Grassmannian), then project x_S onto it and normalise.

    Returns:
        delta_stars: ndarray of shape (n_vocab, C). Unit-norm random direction
            per attractor (or NaN if projection is zero).
    """
    if rng is None:
        rng = np.random.default_rng()

    X = build_attractor_vectors(vocabulary, C=C)
    n_vocab = len(vocabulary)
    delta_stars = np.full((n_vocab, C), np.nan)

    # Random M-dimensional subspace: columns of Q from QR of random (C, M) matrix
    W = rng.standard_normal((C, M))
    Q, _ = np.linalg.qr(W)
    Q = Q[:, :M]
    Pi_rand = Q @ Q.T

    for v_idx, x_S in enumerate(X):
        proj = Pi_rand @ x_S
        norm = np.linalg.norm(proj)
        if norm > 1e-10:
            delta_stars[v_idx] = proj / norm

    return delta_stars


def compute_random_per_attractor_gamma(
    M: int,
    vocabulary: list[tuple[int, ...]],
    C: int = 18,
    k: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a random subspace and compute per-attractor γ and targeting directions.

    Returns:
        (gammas, delta_stars, Pi): per-attractor margins, targeting directions, projection matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    W = rng.standard_normal((C, M))
    Q, _ = np.linalg.qr(W)
    Q = Q[:, :M]
    Pi = Q @ Q.T

    gammas = compute_per_attractor_gamma(Pi, vocabulary, C=C, k=k)

    X = build_attractor_vectors(vocabulary, C=C)
    n_vocab = len(vocabulary)
    delta_stars = np.full((n_vocab, C), np.nan)
    for v_idx, x_S in enumerate(X):
        proj = Pi @ x_S
        norm = np.linalg.norm(proj)
        if norm > 1e-10:
            delta_stars[v_idx] = proj / norm

    return gammas, delta_stars, Pi


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def generate_experiment_configs(
    sdp_results: dict,
    vocabulary: list[tuple[int, ...]],
    per_attractor_gammas_opt: np.ndarray,
    random_results: dict,
    base_config: Path,
    config_dir: Path,
    alpha: float,
    C: int = 18,
) -> list[Path]:
    """Generate validation YAML configs for both optimized and random directions.

    Config naming:
        snn_validate_M{M}_{label}.yaml          — SDP-optimized
        snn_validate_M{M}_rand{seed}_{label}.yaml — random direction
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # --- Optimized configs ---
    for m_idx, M in enumerate(sdp_results["M_values"]):
        Pi = sdp_results["Pi_matrices"][m_idx]
        X = build_attractor_vectors(vocabulary, C=C)
        for v_idx, S0 in enumerate(vocabulary):
            try:
                delta = compute_targeting_direction(Pi, X[v_idx])
            except ValueError:
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

    # --- Random configs ---
    for seed_idx, rand_entry in enumerate(random_results["entries"]):
        M = rand_entry["M"]
        seed = rand_entry["seed"]
        delta_stars = rand_entry["delta_stars"]
        for v_idx, S0 in enumerate(vocabulary):
            delta = delta_stars[v_idx]
            if np.any(np.isnan(delta)):
                continue

            label = "_".join(str(c) for c in S0)
            out_path = config_dir / f"snn_validate_M{M}_rand{seed}_{label}.yaml"
            write_validation_yaml(
                base_config_path=base_config,
                output_path=out_path,
                delta_star=delta,
                label=f"rand{seed}_{label}",
                M=M,
                C=C,
            )
            written.append(out_path)

    print(f"Wrote {len(written)} configs → {config_dir}")
    return written


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_experiment_outputs(
    out_dir: Path,
    vocabulary: list[tuple[int, ...]],
    sdp_results: dict,
    per_attractor_gammas_opt: np.ndarray,
    random_results: dict,
) -> None:
    """Save all experiment outputs to disk.

    Files written:
        vocabulary.json                 — attractor tuples
        sdp_outputs.npz                — M_values, gamma_opt, Pi_matrices, delta_stars
        per_attractor_gammas_opt.npz    — shape (n_M, n_vocab): γ(S; M) per attractor
        random_outputs.npz              — random direction data: gammas, delta_stars, seeds
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Vocabulary
    vocab_path = out_dir / "vocabulary.json"
    with open(vocab_path, "w") as f:
        json.dump([list(s) for s in vocabulary], f, indent=2)

    # SDP outputs (reuse existing format)
    delta_stars_opt = np.full(
        (len(sdp_results["M_values"]), len(vocabulary), len(vocabulary[0]) and 18),
        np.nan,
    )
    X = build_attractor_vectors(vocabulary, C=18)
    for m_idx, Pi in enumerate(sdp_results["Pi_matrices"]):
        for v_idx, x_S in enumerate(X):
            try:
                delta_stars_opt[m_idx, v_idx] = compute_targeting_direction(Pi, x_S)
            except ValueError:
                pass

    npz_path = out_dir / "sdp_outputs.npz"
    np.savez(
        npz_path,
        M_values=np.array(sdp_results["M_values"], dtype=np.int32),
        gamma_opt=np.array(sdp_results["gamma_opt"], dtype=np.float64),
        Pi_matrices=np.array(sdp_results["Pi_matrices"], dtype=np.float64),
        is_isotropic=np.array(sdp_results["is_isotropic"], dtype=bool),
        delta_stars=delta_stars_opt,
        vocabulary=np.array([list(s) for s in vocabulary], dtype=np.int32),
    )

    # Per-attractor gammas (optimized)
    np.savez(
        out_dir / "per_attractor_gammas_opt.npz",
        M_values=np.array(sdp_results["M_values"], dtype=np.int32),
        gammas=per_attractor_gammas_opt,  # shape (n_M, n_vocab)
    )

    # Random outputs
    rand_M = []
    rand_seeds = []
    rand_gammas = []
    rand_deltas = []
    for entry in random_results["entries"]:
        rand_M.append(entry["M"])
        rand_seeds.append(entry["seed"])
        rand_gammas.append(entry["gammas"])
        rand_deltas.append(entry["delta_stars"])

    np.savez(
        out_dir / "random_outputs.npz",
        M_values=np.array(rand_M, dtype=np.int32),
        seeds=np.array(rand_seeds, dtype=np.int32),
        gammas=np.array(rand_gammas, dtype=np.float64),       # (n_entries, n_vocab)
        delta_stars=np.array(rand_deltas, dtype=np.float64),   # (n_entries, n_vocab, C)
    )

    print(f"Saved outputs → {out_dir}")
    print(f"  per_attractor_gammas_opt: {per_attractor_gammas_opt.shape}")
    print(f"  random entries: {len(random_results['entries'])}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Setup for unified γ→π and single-mode experiment."
    )
    parser.add_argument(
        "--base-config",
        default="configs/snn_long_run_perturbed.yaml",
        help="Template YAML config.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/gamma_pi_experiment",
        help="Directory for generated validation YAML configs.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/gamma_pi_setup",
        help="Directory for SDP + random outputs.",
    )
    parser.add_argument("--C", type=int, default=18, help="Number of clusters.")
    parser.add_argument("--k", type=int, default=3, help="Active clusters per attractor.")
    parser.add_argument("--M-min", type=int, default=1, help="Minimum M (default: 1).")
    parser.add_argument("--M-max", type=int, default=12, help="Maximum M (inclusive).")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.25,
        help="Perturbation strength α (embedded in config metadata; default: 0.25).",
    )
    parser.add_argument(
        "--n-random",
        type=int,
        default=3,
        help="Number of random subspace seeds per M value (default: 3).",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=256,
        help="Base RNG seed for random directions (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    C, k = args.C, args.k
    M_range = range(args.M_min, args.M_max + 1)

    base_config = ROOT / args.base_config
    config_dir = ROOT / args.config_dir
    out_dir = ROOT / args.out_dir

    # 1. Build vocabulary (minimal saturating set)
    vocabulary = build_saturating_vocabulary(C=C, k=k)
    print(f"Vocabulary: {len(vocabulary)} attractors (saturating set)")

    # 2. Run SDP for each M
    sdp_results = run_sdp(vocabulary, M_range=M_range, C=C, k=k)

    # 3. Compute per-attractor γ(S; M) for optimized subspaces
    print("\nComputing per-attractor margins (optimized)...")
    n_M = len(sdp_results["M_values"])
    n_vocab = len(vocabulary)
    per_attractor_gammas_opt = np.full((n_M, n_vocab), np.nan)

    for m_idx, (M, Pi) in enumerate(
        zip(sdp_results["M_values"], sdp_results["Pi_matrices"])
    ):
        gammas = compute_per_attractor_gamma(Pi, vocabulary, C=C, k=k)
        per_attractor_gammas_opt[m_idx] = gammas
        n_targetable = np.sum(gammas > 0)
        print(f"  M={M:2d}: γ_min={gammas.min():.4f}  γ_max={gammas.max():.4f}"
              f"  coverage={n_targetable}/{n_vocab}")

    # 4. Generate random directions for each M
    print(f"\nGenerating random directions ({args.n_random} seeds per M)...")
    random_results: dict = {"entries": []}

    for M in sdp_results["M_values"]:
        for seed_offset in range(args.n_random):
            seed = args.random_seed + seed_offset
            rng = np.random.default_rng(seed * 1000 + M)
            gammas, delta_stars, Pi = compute_random_per_attractor_gamma(
                M, vocabulary, C=C, k=k, rng=rng,
            )
            n_targetable = np.sum(gammas > 0)
            print(f"  M={M:2d} seed={seed}: γ_min={gammas.min():.4f}"
                  f"  coverage={n_targetable}/{n_vocab}")
            random_results["entries"].append({
                "M": M,
                "seed": seed,
                "gammas": gammas,
                "delta_stars": delta_stars,
            })

    # 5. Generate configs
    print(f"\nGenerating configs...")
    generate_experiment_configs(
        sdp_results=sdp_results,
        vocabulary=vocabulary,
        per_attractor_gammas_opt=per_attractor_gammas_opt,
        random_results=random_results,
        base_config=base_config,
        config_dir=config_dir,
        alpha=args.alpha,
        C=C,
    )

    # 6. Save outputs
    save_experiment_outputs(
        out_dir=out_dir,
        vocabulary=vocabulary,
        sdp_results=sdp_results,
        per_attractor_gammas_opt=per_attractor_gammas_opt,
        random_results=random_results,
    )

    # 7. Print instructions
    print(f"""
{'='*60}
Setup complete.

To run the simulations:
    python scripts/snn_long_runs/run_capacity_validation.py \\
        --config-dir {args.config_dir} \\
        --save-dir simulations/gamma_pi_validation \\
        --range 0 0.5 2 \\
        --n-repeats 75 \\
        --parallel --executor process --max-workers 16 \\
        --lite-output --no-plots --skip-existing

To analyse results:
    python analyze_gamma_pi_experiment.py \\
        --validation-dir simulations/gamma_pi_validation \\
        --setup-dir {args.out_dir}
{'='*60}
""")


if __name__ == "__main__":
    main()
