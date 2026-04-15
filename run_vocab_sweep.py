#!/usr/bin/env python3
"""Sweep vocabulary size and compute SDP optimal margin Γ_min(W*; M) for each M.

For each vocabulary size in a geometric grid from 12 to 816 (the full H_k space
for C=18, k=3), solve the SDP and print a table of γ vs vocab_size per M.

Usage:
    python run_vocab_sweep.py
    python run_vocab_sweep.py --M-min 1 --M-max 18
    python run_vocab_sweep.py --out results/vocab_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from run_capacity_experiment import build_saturating_vocabulary
from neuro_mod.analysis.capacity.sdp import capacity_curve


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
    """Build a vocabulary of size n that greedily maximises intra-T single-swap pairs.

    At each step the attractor with the most swap-neighbours already in T is added.
    Ties are broken by total degree in the swap graph (most potential future edges).
    The first seed is the highest-degree node in the swap graph.
    """
    pool_set = set(pool)
    # Precompute swap neighbours within pool
    neighbors: dict[tuple, list[tuple]] = {
        s: [nb for nb in single_swap_neighbors(s, C) if nb in pool_set]
        for s in pool
    }
    degree = {s: len(nb) for s, nb in neighbors.items()}

    T: set[tuple] = set()
    remaining: set[tuple] = set(pool)
    # count of neighbours each candidate has inside T
    nb_in_T: dict[tuple, int] = {s: 0 for s in pool}

    # Seed: highest-degree node
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


def geometric_sizes(lo: int, hi: int, n_points: int = 20) -> list[int]:
    """Return ~n_points integers spaced geometrically between lo and hi (inclusive)."""
    import math
    sizes = set()
    sizes.add(lo)
    sizes.add(hi)
    for i in range(1, n_points - 1):
        v = int(round(lo * (hi / lo) ** (i / (n_points - 1))))
        sizes.add(max(lo, min(hi, v)))
    return sorted(sizes)


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
        "--dense",
        action="store_true",
        default=False,
        help=(
            "Build vocabularies greedily to maximise intra-T single-swap pairs "
            "(swap-dense construction) instead of the saturating-seeded lex pool."
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
    M_range = range(args.M_min, args.M_max + 1)

    saturating = build_saturating_vocabulary(C=C, k=k)
    pool = all_attractors(C=C, k=k)  # 816 attractors for C=18, k=3
    min_size = len(saturating)
    if args.pool_size == -1:
        max_size = len(pool)
    else:
        max_size = min(args.pool_size, len(pool))

    sizes = geometric_sizes(min_size, max_size, n_points=args.n_sizes)
    mode = "dense" if args.dense else "saturating-seeded"
    print(f"C={C}, k={k}, |H_k|={max_size}, saturating={min_size}, mode={mode}")
    print(f"Vocabulary sizes: {sizes}")
    print(f"M range: {list(M_range)}\n")

    # Header
    M_list = list(M_range)
    header = ["vocab_size"] + [f"M={m}" for m in M_list]
    rows: list[list] = []

    for size in sizes:
        if args.dense:
            vocab = build_dense_vocabulary(size, pool, C=C, k=k)
        else:
            vocab = build_vocabulary_from_pool(size, pool, saturating)
        assert len(vocab) == size, f"Expected {size} attractors, got {len(vocab)}"

        results = capacity_curve(vocab, M_range=M_range, C=C, k=k)
        gammas = results["gamma_opt"]

        row = [size] + [f"{g:.4f}" for g in gammas]
        rows.append(row)

        gamma_str = "  ".join(f"M={m}: {g:.4f}" for m, g in zip(M_list, gammas))
        print(f"vocab={size:4d}  {gamma_str}")

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
