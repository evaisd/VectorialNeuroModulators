"""Generate YAML configuration files for basis sweeps and validation experiments.

Usage (basis sweeps):
    python generate_validation_configs.py basis --base-config configs/snn_long_run_perturbed.yaml
                                                --output-dir configs/basis_sweeps/

Usage (validation configs):
    python generate_validation_configs.py validation --base-config configs/snn_long_run_perturbed.yaml
                                                     --output-dir configs/validation/
                                                     --delta-star path/to/delta_star_M3_S0.npy
                                                     --label M3_S0_0_5_12
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False


def _require_yaml() -> None:
    if not _YAML_AVAILABLE:
        raise ImportError("PyYAML is required. Install with: pip install pyyaml")


def _load_yaml(path: Path) -> dict:
    _require_yaml()
    with open(path) as f:
        return yaml.safe_load(f)


def _save_yaml(config: dict, path: Path) -> None:
    _require_yaml()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Basis sweep configs (one per cluster)
# ---------------------------------------------------------------------------

def generate_basis_configs(
    base_config_path: Path | str,
    output_dir: Path | str,
    C: int = 18,
    skip_existing: bool = True,
) -> list[Path]:
    """Generate 18 single-cluster basis sweep YAML configs.

    Each config is identical to the base config except the perturbation
    block is replaced with a single-cluster rate perturbation:
        vectors: [[0, ..., 1, ..., 0]]  (1 at position c, 0 elsewhere)
        involved_clusters: [[c]]
        params: [1.0]

    The sweep infrastructure varies params[0] as the scale α.

    Args:
        base_config_path: Path to a baseline perturbed config (used as template).
            Typically snn_long_run_perturbed.yaml.
        output_dir: Directory to write snn_basis_c{c}.yaml files.
        C: Number of clusters (default 18).
        skip_existing: If True, skip writing configs that already exist.

    Returns:
        List of written (or skipped) config paths.
    """
    base_config_path = Path(base_config_path)
    output_dir = Path(output_dir)
    base_config = _load_yaml(base_config_path)

    written = []
    for c in range(C):
        out_path = output_dir / f"snn_basis_c{c}.yaml"
        if skip_existing and out_path.exists():
            written.append(out_path)
            continue

        config = copy.deepcopy(base_config)

        # Build e_c vector: 1 at position c, 0 elsewhere
        e_c = [0] * C
        e_c[c] = 1

        config["perturbation"] = {
            "rate": {
                "vectors": [e_c],
                "involved_clusters": [[c]],
                "seed": 256,
                "params": [1.0],
            }
        }

        _save_yaml(config, out_path)
        written.append(out_path)

    return written


# ---------------------------------------------------------------------------
# Validation sweep configs (one per (M, S0) pair)
# ---------------------------------------------------------------------------

def write_validation_yaml(
    base_config_path: Path | str,
    output_path: Path | str,
    delta_star: np.ndarray,
    label: str,
    M: int,
    C: int = 18,
) -> Path:
    """Write a validation sweep config using delta_star as the perturbation vector.

    The config sweeps the scalar coefficient α (perturbation.rate.params[0])
    while holding the perturbation direction fixed at delta_star. The network
    receives the perturbation α · delta_star at each sweep step.

    Args:
        base_config_path: Path to a baseline config (template).
        output_path: Where to write the new YAML.
        delta_star: ndarray of shape (C,) — unit-norm targeting direction.
        label: Human-readable label (e.g. 'M3_S0_0_5_12') embedded in sim_name.
        M: Number of modes used to derive delta_star (for documentation).
        C: Number of clusters.

    Returns:
        Path to the written config.
    """
    base_config_path = Path(base_config_path)
    output_path = Path(output_path)

    if len(delta_star) != C:
        raise ValueError(
            f"delta_star has length {len(delta_star)}, expected C={C}."
        )

    config = copy.deepcopy(_load_yaml(base_config_path))

    # Store full delta_star as the single basis vector
    delta_list = [float(x) for x in delta_star]
    involved = list(range(C))  # all clusters may have non-zero components

    config["perturbation"] = {
        "rate": {
            "vectors": [delta_list],
            "involved_clusters": [involved],
            "seed": 256,
            "params": [1.0],  # swept as perturbation.rate.params
        }
    }

    # Embed metadata in settings
    if "settings" not in config:
        config["settings"] = {}
    config["settings"]["sim_name"] = f"validate_{label}_M{M}"

    _save_yaml(config, output_path)
    return output_path


def generate_validation_configs(
    base_config_path: Path | str,
    output_dir: Path | str,
    targeting_results: dict,
    vocabulary_info: list[dict],
    M_values: list[int] | None = None,
    skip_existing: bool = True,
) -> list[Path]:
    """Generate all validation sweep configs from SDP targeting results.

    Args:
        base_config_path: Template config path.
        output_dir: Directory for output configs.
        targeting_results: Output of sdp.capacity_curve — dict with keys
            'M_values', 'Pi_matrices'. Used to compute delta_star per (M, S0).
        vocabulary_info: Output of vocabulary.classify_vocabulary_difficulty —
            list of dicts with keys 'attractor', 'difficulty'.
        M_values: Which M values to generate configs for. If None, uses all M
            in targeting_results.
        skip_existing: If True, skip writing configs that already exist.

    Returns:
        List of written config paths.
    """
    from neuro_mod.analysis.capacity.sdp import (
        build_attractor_vectors,
        compute_targeting_direction,
    )

    base_config_path = Path(base_config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if M_values is None:
        M_values = targeting_results["M_values"]

    # Index Pi matrices by M
    Pi_by_M = dict(zip(targeting_results["M_values"], targeting_results["Pi_matrices"]))

    written = []
    for entry in vocabulary_info:
        attractor = entry["attractor"]
        difficulty = entry["difficulty"]
        C = len(Pi_by_M[M_values[0]])

        x_S0 = np.zeros(C)
        for c in attractor:
            x_S0[c] = 1.0

        label_base = "_".join(str(c) for c in attractor)

        for M in M_values:
            Pi = Pi_by_M.get(M)
            if Pi is None:
                continue

            try:
                delta_star = compute_targeting_direction(Pi, x_S0)
            except ValueError:
                continue

            label = f"{difficulty}_{label_base}"
            out_path = output_dir / f"snn_validate_M{M}_{label}.yaml"

            if skip_existing and out_path.exists():
                written.append(out_path)
                continue

            write_validation_yaml(
                base_config_path=base_config_path,
                output_path=out_path,
                delta_star=delta_star,
                label=label,
                M=M,
                C=C,
            )
            written.append(out_path)

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate basis sweep and validation YAML configs."
    )
    subparsers = parser.add_subparsers(dest="command")

    basis_parser = subparsers.add_parser("basis", help="Generate basis sweep configs.")
    basis_parser.add_argument("--base-config", required=True)
    basis_parser.add_argument("--output-dir", required=True)
    basis_parser.add_argument("--C", type=int, default=18)

    val_parser = subparsers.add_parser("validation", help="Generate validation configs.")
    val_parser.add_argument("--base-config", required=True)
    val_parser.add_argument("--output-dir", required=True)
    val_parser.add_argument("--delta-star", required=True,
                            help="Path to .npy file containing delta_star (shape C,).")
    val_parser.add_argument("--label", required=True)
    val_parser.add_argument("--M", type=int, required=True)
    val_parser.add_argument("--C", type=int, default=18)

    args = parser.parse_args()

    if args.command == "basis":
        paths = generate_basis_configs(
            base_config_path=args.base_config,
            output_dir=args.output_dir,
            C=args.C,
        )
        print(f"Wrote {len(paths)} basis configs to {args.output_dir}")

    elif args.command == "validation":
        delta_star = np.load(args.delta_star)
        path = write_validation_yaml(
            base_config_path=args.base_config,
            output_path=Path(args.output_dir) / f"snn_validate_{args.label}_M{args.M}.yaml",
            delta_star=delta_star,
            label=args.label,
            M=args.M,
            C=args.C,
        )
        print(f"Wrote validation config to {path}")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
