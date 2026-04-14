"""Capacity analysis: SDP optimisation, sensitivity maps, and leakage characterisation."""

from neuro_mod.analysis.capacity.sdp import (
    build_attractor_vectors,
    build_difference_vectors,
    build_minimal_saturating_vocabulary,
    capacity_curve,
    compute_targeting_direction,
    solve_subspace_sdp,
)
from neuro_mod.analysis.capacity.sensitivity import (
    build_G_matrix,
    load_attractors_from_npy,
    load_sweep_probabilities,
    predict_probabilities_from_G,
    validate_linearity,
)
from neuro_mod.analysis.capacity.vocabulary import (
    check_saturation_coverage,
    select_vocabulary_from_empirical,
)
from neuro_mod.analysis.capacity.leakage import (
    classify_attractor_role,
    compute_leakage_profile,
    find_metastable_boundary,
)

__all__ = [
    # sdp
    "build_attractor_vectors",
    "build_difference_vectors",
    "build_minimal_saturating_vocabulary",
    "capacity_curve",
    "compute_targeting_direction",
    "solve_subspace_sdp",
    # sensitivity
    "build_G_matrix",
    "load_attractors_from_npy",
    "load_sweep_probabilities",
    "predict_probabilities_from_G",
    "validate_linearity",
    # vocabulary
    "check_saturation_coverage",
    "select_vocabulary_from_empirical",
    # leakage
    "classify_attractor_role",
    "compute_leakage_profile",
    "find_metastable_boundary",
]
