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

    Returns (vocab_size, M, gamma).
    """
    vocab, M, C, k = args
    X = build_attractor_vectors(vocab, C=C)
    diff_vecs = build_difference_vectors(X, list(range(len(vocab))), k=k, bottleneck_only=True)
    if not diff_vecs:
        return (len(vocab), M, 1.)
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
        "--out",
        default=None,
        help="Optional path to write CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    C, k = args.C, args.k
    M_list = list(range(args.M_min, args.M_max + 1))

    saturating = build_saturating_vocabulary(C=C, k=k)
    pool = all_attractors(C=C, k=k)
    min_size = len(saturating)
    max_size = len(pool) if args.pool_size == -1 else min(args.pool_size, len(pool))

    if args.step is not None:
        sizes = list(range(min_size, max_size + 1, args.step))
        if sizes[-1] != max_size:
            sizes.append(max_size)
    else:
        sizes = geometric_sizes(min_size, max_size, n_points=args.n_sizes)
    mode = "dense" if args.dense else "sparse" if args.sparse else "random" if args.random else "saturating-seeded"
    print(f"C={C}, k={k}, |H_k|={max_size}, saturating={min_size}, mode={mode}")
    print(f"Vocabulary sizes: {sizes}")
    print(f"M range: {M_list}  |  workers: {args.jobs}\n")

    # For random mode, shuffle the pool once and slice — guarantees monotonicity.
    if args.random:
        rng = np.random.default_rng(args.seed)
        shuffled_pool = list(pool[:max_size])
        rng.shuffle(shuffled_pool)
        random_vocabs = {size: sorted(shuffled_pool[:size]) for size in sizes}

    def build_vocab(size: int) -> list[tuple]:
        if args.dense:
            return build_dense_vocabulary(size, pool, C=C, k=k)
        elif args.sparse:
            return build_sparse_vocabulary(size, pool, C=C, k=k)
        elif args.random:
            return random_vocabs[size]
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
            vocab = build_vocab(size)
            assert len(vocab) == size

            # Dispatch all M values for this size in parallel
            futures = {
                executor.submit(_solve_one, (vocab, M, C, k)): M
                for M in M_list
            }
            size_results: dict[int, float] = {}
            for future in as_completed(futures):
                _, M, gamma = future.result()
                size_results[M] = gamma

            gammas = [size_results[M] for M in M_list]
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
