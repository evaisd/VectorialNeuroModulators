"""Target vocabulary selection for the controllability SDP.

A vocabulary T is a set of attractor identities used to define the SDP
constraints. For the SDP to be meaningful T must satisfy the saturation
coverage condition: every ordered cluster pair (i→j) must be covered by
some S ∈ T with i ∈ S and j ∉ S.

For C=18, k=3 the theoretical minimum is |T|_min = 12.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def check_saturation_coverage(
    vocabulary: list[tuple[int, ...]],
    C: int = 18,
) -> dict:
    """Check whether a vocabulary satisfies the saturation coverage condition.

    Coverage condition: for every ordered pair (i, j) with i ≠ j,
    there exists S ∈ T such that i ∈ S and j ∉ S.

    Args:
        vocabulary: List of attractor tuples.
        C: Total number of clusters.

    Returns:
        dict with keys:
            'is_saturating': bool — True if all C*(C-1) directed pairs covered
            'n_covered':     int  — number of covered directed pairs
            'n_total':       int  — C*(C-1) total directed pairs
            'uncovered':     list[tuple[int,int]] — uncovered (i, j) pairs
    """
    covered = set()
    for S in vocabulary:
        S_set = set(S)
        for i in S_set:
            for j in range(C):
                if j not in S_set:
                    covered.add((i, j))

    all_pairs = {(i, j) for i in range(C) for j in range(C) if i != j}
    uncovered = sorted(all_pairs - covered)

    return {
        "is_saturating": len(uncovered) == 0,
        "n_covered": len(covered),
        "n_total": len(all_pairs),
        "uncovered": uncovered,
    }


# ---------------------------------------------------------------------------
# Vocabulary selection from empirical attractors
# ---------------------------------------------------------------------------

def select_vocabulary_from_empirical(
    attractor_tuples: list[tuple[int, ...]],
    probabilities: np.ndarray,
    C: int = 18,
    k: int = 3,
    n_targets: int = 12,
    strategy: str = "greedy_saturating",
) -> list[tuple[int, ...]]:
    """Select a target vocabulary from empirically observed attractors.

    Strategies:
        'greedy_saturating':
            Greedily add the highest-probability k-hot attractor that
            contributes at least one new directed pair (i→j) to the covered
            set, until coverage is saturated or n_targets is reached.
        'top_k':
            Simply take the n_targets highest-probability k-hot attractors,
            regardless of coverage.

    Args:
        attractor_tuples: All observed attractor identities.
        probabilities: Corresponding steady-state probabilities
            (same order as attractor_tuples).
        C: Number of clusters.
        k: Required attractor size (only k-hot attractors considered).
        n_targets: Maximum vocabulary size.
        strategy: Selection method.

    Returns:
        Selected vocabulary as a list of tuples, ordered by selection priority.

    Raises:
        ValueError: If fewer than n_targets k-hot attractors are available.
    """
    # Filter to k-hot attractors
    k_hot = [
        (tup, p)
        for tup, p in zip(attractor_tuples, probabilities)
        if len(tup) == k
    ]
    if not k_hot:
        raise ValueError(f"No k={k}-hot attractors found in the provided list.")

    # Sort by descending probability
    k_hot.sort(key=lambda x: -x[1])

    if strategy == "top_k":
        return [tup for tup, _ in k_hot[:n_targets]]

    if strategy == "greedy_saturating":
        covered: set[tuple[int, int]] = set()
        all_pairs = {(i, j) for i in range(C) for j in range(C) if i != j}
        vocabulary: list[tuple[int, ...]] = []

        for tup, _ in k_hot:
            if len(vocabulary) >= n_targets:
                break
            # New pairs this attractor covers
            S_set = set(tup)
            new_pairs = {
                (i, j)
                for i in S_set
                for j in range(C)
                if j not in S_set and (i, j) not in covered
            }
            if new_pairs or len(vocabulary) < n_targets:
                vocabulary.append(tup)
                covered.update(new_pairs)
            if covered >= all_pairs:
                break

        return vocabulary

    raise ValueError(
        f"Unknown strategy '{strategy}'. Choose 'greedy_saturating' or 'top_k'."
    )


def classify_vocabulary_difficulty(
    vocabulary: list[tuple[int, ...]],
    attractor_tuples: list[tuple[int, ...]],
    probabilities: np.ndarray,
    C: int = 18,
) -> list[dict]:
    """Classify each vocabulary attractor as easy / medium / hard to target.

    Difficulty is determined by the number of bottleneck competitors
    (attractors in the full universe differing by exactly one cluster swap):
        easy:   0–2 bottleneck competitors in the observed vocabulary
        medium: 3–5 bottleneck competitors
        hard:   ≥6 bottleneck competitors

    Args:
        vocabulary: The selected target vocabulary.
        attractor_tuples: Full list of observed attractors.
        probabilities: Corresponding probabilities (same order).
        C: Number of clusters.

    Returns:
        List of dicts (one per vocabulary attractor) with keys:
            'attractor':            tuple
            'baseline_prob':        float
            'n_bottleneck_competitors': int
            'difficulty':           str — 'easy', 'medium', or 'hard'
    """
    prob_map = dict(zip(attractor_tuples, probabilities))
    results = []

    for S in vocabulary:
        S_set = set(S)
        n_bottleneck = 0
        for S_prime in attractor_tuples:
            if S_prime == S:
                continue
            sym_diff = len(set(S) ^ set(S_prime))
            if sym_diff == 2:
                n_bottleneck += 1

        if n_bottleneck <= 2:
            difficulty = "easy"
        elif n_bottleneck <= 5:
            difficulty = "medium"
        else:
            difficulty = "hard"

        results.append({
            "attractor": S,
            "baseline_prob": prob_map.get(S, np.nan),
            "n_bottleneck_competitors": n_bottleneck,
            "difficulty": difficulty,
        })

    return results
