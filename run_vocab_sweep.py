#!/usr/bin/env python3
"""Sweep vocabulary size and compute SDP optimal margin Γ_min(W*; M) for each M.

For each vocabulary size in a geometric grid from 12 to 816 (the full H_k space
for C=18, k=3), solve the SDP and print a table of γ vs vocab_size per M.

Usage:
    python run_vocab_sweep.py
    python run_vocab_sweep.py --M-min 1 --M-max 18
    python run_vocab_sweep.py --dense --out results/sweep_dense.csv
    python run_vocab_sweep.py --sparse --out results/sweep_sparse.csv
    python run_vocab_sweep.py --jobs 8 --out results/vocab_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from run_capacity_experiment import build_saturating_vocabulary
from neuro_mod.analysis.capacity.sdp import (
    build_attractor_vectors,
    build_difference_vectors,
    solve_subspace_sdp,
)


# ---------------------------------------------------------------------------
# Vocabulary construction helpers
# ---------------------------------------------------------------------------

def all_attractors(C: int = 18, k: int = 3) -> list[tuple[int, ...]]:
    return list(combinations(range(C), k))


def build_vocabulary_from_pool(
    n: int,
    pool: list[tuple[int, ...]],
    saturating: list[tuple[int, ...]],
) -> list[tuple[int, ...]]:
    """Return a vocabulary of size n that always includes the saturating set."""
    sat_set = set(saturating)
    extras = [v for v in pool if v not in sat_set]
    vocab = sorted(saturating + extras[: n - len(saturating)])
    return vocab


def single_swap_neighbors(s: tuple, C: int) -> list[tuple]:
    """Return all attractors reachable from s by a single cluster swap."""
    s_set = set(s)
    neighbors = []
    for i in s_set:
        for j in range(C):
            if j not in s_set:
                neighbors.append(tuple(sorted((s_set - {i}) | {j})))
    return neighbors


def build_dense_vocabulary(
    n: int,
    pool: list[tuple[int, ...]],
    C: int,
    k: int,
) -> list[tuple[int, ...]]:
    """Greedily maximise intra-T single-swap pairs (hardest vocab)."""
    pool_set = set(pool)
    neighbors: dict[tuple, list[tuple]] = {
        s: [nb for nb in single_swap_neighbors(s, C) if nb in pool_set]
        for s in pool
    }
    degree = {s: len(nb) for s, nb in neighbors.items()}

    T: set[tuple] = set()
    remaining: set[tuple] = set(pool)
    nb_in_T: dict[tuple, int] = {s: 0 for s in pool}

    seed = max(pool, key=lambda s: degree[s])
    T.add(seed)
    remaining.remove(seed)
    for nb in neighbors[seed]:
        nb_in_T[nb] += 1

    while len(T) < n and remaining:
        best = max(remaining, key=lambda s: (nb_in_T[s], degree[s]))
        T.add(best)
        remaining.remove(best)
        for nb in neighbors[best]:
            if nb in remaining:
                nb_in_T[nb] += 1

    return sorted(T)


def build_sparse_vocabulary(
    n: int,
    pool: list[tuple[int, ...]],
    C: int,
    k: int,
) -> list[tuple[int, ...]]:
    """Greedily minimise intra-T single-swap pairs (easiest vocab)."""
    pool_set = set(pool)
    neighbors: dict[tuple, list[tuple]] = {
        s: [nb for nb in single_swap_neighbors(s, C) if nb in pool_set]
        for s in pool
    }
    degree = {s: len(nb) for s, nb in neighbors.items()}

    T: set[tuple] = set()
    remaining: set[tuple] = set(pool)
    nb_in_T: dict[tuple, int] = {s: 0 for s in pool}

    seed = min(pool, key=lambda s: (degree[s], s))
    T.add(seed)
    remaining.remove(seed)
    for nb in neighbors[seed]:
        nb_in_T[nb] += 1

    while len(T) < n and remaining:
        best = min(remaining, key=lambda s: (nb_in_T[s], degree[s], s))
        T.add(best)
        remaining.remove(best)
        for nb in neighbors[best]:
            if nb in remaining:
                nb_in_T[nb] += 1

    return sorted(T)



def geometric_sizes(lo: int, hi: int, n_points: int = 20) -> list[int]:
    """Return ~n_points integers spaced geometrically between lo and hi (inclusive)."""
    sizes = set()
    sizes.add(lo)
    sizes.add(hi)
    for i in range(1, n_points - 1):
        v = int(round(lo * (hi / lo) ** (i / (n_points - 1))))
        sizes.add(max(lo, min(hi, v)))
    return sorted(sizes)


# ---------------------------------------------------------------------------
# Parallelisable worker — one SDP solve per (vocab, M)
# ---------------------------------------------------------------------------

def _solve_one(args: tuple) -> tuple[int, int, float]:
    """Worker: solve SDP for one (vocab_size, M) pair.

    Difference vectors are generated for each vocabulary member against ALL
    single-swap neighbors in the full H_k (not just intra-vocabulary pairs),
    then deduplicated by directed cluster-pair (i -> j).  This gives a margin
    that is physically bounded by sqrt(2/k) and removes the empty-constraint
    sentinel that inflated sparse-vocabulary results.

    Returns (vocab_size, M, gamma).
    """
    vocab, M, C, k = args
    # Collect unique directed pairs (i -> j) activated by the vocabulary:
    # pair (i -> j) is activated by attractor S when i in S and j not in S.
    seen: set[tuple[int, int]] = set()
    diff_vecs: list[tuple[int, int, np.ndarray]] = []
    scale = 1.0 / np.sqrt(k)
    for s in vocab:
        s_set = set(s)
        for i in s_set:
            for j in range(C):
                if j not in s_set and (i, j) not in seen:
                    seen.add((i, j))
                    d = np.zeros(C)
                    d[i] = scale
                    d[j] = -scale
                    diff_vecs.append((i, j, d))
    result = solve_subspace_sdp(diff_vecs, M=M, C=C)
    return (len(vocab), M, result["gamma"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Vocab-size sweep of SDP capacity curve.")
    parser.add_argument("--C", type=int, default=18)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--M-min", type=int, default=1)
    parser.add_argument("--M-max", type=int, default=18)
    parser.add_argument("--pool-size", type=int, default=-1)
    parser.add_argument(
        "--n-sizes",
        type=int,
        default=20,
        help="Number of vocabulary sizes to sample (geometric grid).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Fixed arithmetic step between vocabulary sizes (overrides --n-sizes).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers (default: all CPUs).",
    )
    parser.add_argument(
        "--converge",
        action="store_true",
        default=False,
        help="Stop early when γ values stabilise.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Number of consecutive identical rows (to 4 dp) required to declare convergence (default: 2).",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=None,
        help="Minimum vocabulary size that must be reached before --converge can trigger.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--dense",
        action="store_true",
        default=False,
        help="Greedily maximise intra-T single-swap pairs (hardest vocab).",
    )
    mode_group.add_argument(
        "--sparse",
        action="store_true",
        default=False,
        help="Greedily minimise intra-T single-swap pairs (easiest vocab).",
    )
    mode_group.add_argument(
        "--random",
        action="store_true",
        default=False,
        help="Sample vocabulary uniformly at random from the pool.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --random mode.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help=(
            "Number of independent random vocabularies to average over per size "
            "(--random mode only). Seeds are derived from --seed + index."
        ),
    )
    parser.add_argument(
        "--K-max",
        type=int,
        default=None,
        help=(
            "Hard cap on the maximum vocabulary size. Useful for large C where "
            "you want to stay in the sub-saturation regime without biasing the "
            "random sampling (unlike --pool-size, this does not restrict the pool "
            "from which random vocabularies are drawn)."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    C, k = args.C, args.k
    M_list = list(range(args.M_min, args.M_max + 1))

    pool = all_attractors(C=C, k=k)

    # Determine min vocabulary size — use saturating set when available,
    # otherwise fall back to ceil(2C/k) so non-divisible C values work too.
    try:
        saturating = build_saturating_vocabulary(C=C, k=k)
        min_size = len(saturating)
    except ValueError:
        saturating = []
        min_size = (2 * C + k - 1) // k  # ceil(2C/k)

    # max_size: pool-size cap first, then K-max cap.
    max_size = len(pool) if args.pool_size == -1 else min(args.pool_size, len(pool))
    if args.K_max is not None:
        max_size = min(max_size, args.K_max)

    if args.step is not None:
        sizes = list(range(min_size, max_size + 1, args.step))
        if sizes[-1] != max_size:
            sizes.append(max_size)
    else:
        sizes = geometric_sizes(min_size, max_size, n_points=args.n_sizes)
    mode = "dense" if args.dense else "sparse" if args.sparse else "random" if args.random else "saturating-seeded"
    n_seeds = args.n_seeds if args.random else 1
    print(f"C={C}, k={k}, |H_k|={len(pool)}, saturating≈{min_size}, mode={mode}")
    if args.random and n_seeds > 1:
        print(f"Averaging over {n_seeds} random seeds (base seed={args.seed})")
    print(f"Vocabulary sizes: {sizes}")
    print(f"M range: {M_list}  |  workers: {args.jobs}\n")

    # Pre-build all random vocabularies across seeds so workers are stateless.
    # For each (seed_idx, size) we need a distinct random subset of pool.
    if args.random:
        base_seed = args.seed if args.seed is not None else 0
        seed_vocabs: list[dict[int, list[tuple]]] = []
        for seed_idx in range(n_seeds):
            rng = np.random.default_rng(base_seed + seed_idx)
            shuffled = list(pool)  # sample from the full pool (no bias)
            rng.shuffle(shuffled)
            seed_vocabs.append({size: sorted(shuffled[:size]) for size in sizes})

    def build_vocab(size: int, seed_idx: int = 0) -> list[tuple]:
        if args.dense:
            return build_dense_vocabulary(size, pool, C=C, k=k)
        elif args.sparse:
            return build_sparse_vocabulary(size, pool, C=C, k=k)
        elif args.random:
            return seed_vocabs[seed_idx][size]
        else:
            return build_vocabulary_from_pool(size, pool, saturating)

    def is_converged(rows: list[list], patience: int) -> bool:
        """True if the last `patience` rows have identical γ strings (ignores inf rows)."""
        if len(rows) < patience:
            return False
        tail = [r[1:] for r in rows[-patience:]]  # strip vocab_size column
        if any("inf" in str(v) for v in tail[0]):
            return False  # don't converge on unconstrained rows
        return all(r == tail[0] for r in tail[1:])

    header = ["vocab_size"] + [f"M={m}" for m in M_list]
    rows: list[list] = []

    print(f"{'vocab_size':<12}" + "".join(f"{'M='+str(m):>9}" for m in M_list))

    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for size in sizes:
            # Collect one set of gammas per seed, then average.
            seed_gamma_lists: list[dict[int, float]] = []
            for seed_idx in range(n_seeds):
                vocab = build_vocab(size, seed_idx)
                assert len(vocab) == size
                futures = {
                    executor.submit(_solve_one, (vocab, M, C, k)): M
                    for M in M_list
                }
                sg: dict[int, float] = {}
                for future in as_completed(futures):
                    _, M, gamma = future.result()
                    sg[M] = gamma
                seed_gamma_lists.append(sg)

            # Average Γ_min across seeds (NaN-safe: inf values kept as inf).
            gammas: list[float] = []
            for M in M_list:
                vals = [sg[M] for sg in seed_gamma_lists]
                finite = [v for v in vals if not (v != v)]  # filter NaN
                gammas.append(float(np.mean(finite)) if finite else float("nan"))

            row = [size] + [f"{g:.4f}" for g in gammas]
            rows.append(row)
            print(f"{size:<12}" + "".join(f"{g:>9.4f}" for g in gammas))

            if args.converge and (args.min_size is None or size >= args.min_size) and is_converged(rows, args.patience):
                print(f"\nConverged after vocab={size} ({args.patience} identical rows). Stopping.")
                break

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
