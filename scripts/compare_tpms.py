#!/usr/bin/env python3
"""Align transition probability matrices (TPMs) across runs.

Example:
  python scripts/compare_tpms.py results/run_a/processed results/run_b/processed \
    --labels identity --out aligned_tpms
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from neuro_mod.core.spiking_net.analysis import SNNAnalyzer, helpers


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align transition matrices across runs using canonical labels."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to processed run directories (containing attractors.npy).")
    parser.add_argument(
        "--labels",
        choices=["identity", "idx"],
        default="identity",
        help="Label space to use for TPM indices.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output directory to save aligned TPM CSV files.")
    return parser


def _load_tpms(paths: list[Path], labels: str) -> dict[str, pd.DataFrame]:
    tpms: dict[str, pd.DataFrame] = {}
    for path in paths:
        analyzer = SNNAnalyzer(path)
        tpm = analyzer.transitions_matrix(labels=labels)
        if not tpm.empty:
            tpms[path.name] = tpm
    return tpms


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    paths = [Path(p) for p in args.paths]
    tpms = _load_tpms(paths, args.labels)
    if not tpms:
        raise SystemExit("No TPMs found in provided paths.")

    canonical_labels, aligned = helpers.align_transition_matrices(tpms)

    print(f"Aligned {len(aligned)} TPMs; canonical labels: {len(canonical_labels)}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for key, tpm in aligned.items():
            out_path = args.out / f"{key}_tpm.csv"
            tpm.to_csv(out_path, index=True)
        print(f"Saved aligned TPMs to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
