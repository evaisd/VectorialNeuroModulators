#!/usr/bin/env python3
"""Run capacity validation sweeps for all configs in configs/capacity_experiment/.

For each snn_validate_M{M}_{label}.yaml, calls sweep_snn_perturbed.py with a
fixed α-range and saves results to save_dir/M{M}/{label}/.

Usage:
    python scripts/snn_long_runs/run_capacity_validation.py
    python scripts/snn_long_runs/run_capacity_validation.py \\
        --config-dir configs/capacity_experiment \\
        --save-dir simulations/capacity_validation \\
        --range 0 0.1 30 \\
        --n-repeats 1 \\
        --parallel --executor process --max-workers 16 \\
        --lite-output --no-plots --no-compress \\
        --skip-existing
    python scripts/snn_long_runs/run_capacity_validation.py --dry-run --M 4 5
"""

from __future__ import annotations

import argparse
import concurrent.futures
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SWEEP_SCRIPT = Path(__file__).resolve().parent / "sweep_snn_perturbed.py"
PYTHON = sys.executable

_CONFIG_RE = re.compile(r"^snn_validate_(M\d+)_(.+)\.yaml$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_config_name(config_path: Path) -> tuple[str, str] | None:
    """Return (M_label, attractor_label) or None if the name doesn't match."""
    m = _CONFIG_RE.match(config_path.name)
    if m:
        return m.group(1), m.group(2)
    return None


def _sort_key(config_path: Path) -> tuple[int, str]:
    """Sort numerically by M, then lexicographically by attractor label."""
    parts = _parse_config_name(config_path)
    if parts is None:
        return (9999, config_path.name)
    M_label, attractor_label = parts
    return (int(M_label[1:]), attractor_label)


def _is_done(save_dir: Path) -> bool:
    return (save_dir / "dataframes" / "sweep_summary.parquet").exists()


def _build_cmd(
    config_path: Path,
    save_dir: Path,
    *,
    sweep_low: float,
    sweep_high: float,
    sweep_num: int,
    n_repeats: int,
    max_workers: int,
    executor: str,
    log_level: str,
    style: str,
    raster_plots: str,
) -> list[str]:
    return [
        PYTHON, str(SWEEP_SCRIPT),
        "--config", str(config_path),
        "--save-dir", str(save_dir),
        "--range", str(sweep_low), str(sweep_high), str(sweep_num),
        "--n-repeats", str(n_repeats),
        "--parallel",
        "--executor", executor,
        "--max-workers", str(max_workers),
        "--lite-output",
        "--keep-raw",
        "--raster-plots", raster_plots,
        "--no-plots",
        "--no-compress",
        "--log-level", log_level,
        "--style", style,
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orchestrate capacity-validation sweeps over all experiment configs.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/capacity_experiment",
        help="Directory containing snn_validate_*.yaml configs.",
    )
    parser.add_argument(
        "--save-dir",
        default="simulations/capacity_validation",
        help="Output root. Results written to <save-dir>/M{M}/{label}/.",
    )
    parser.add_argument(
        "--range",
        nargs=3,
        type=float,
        metavar=("LOW", "HIGH", "NUM"),
        default=[0.0, 0.1, 30],
        help="Sweep α range (default: 0 0.1 30).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Repeats per sweep point (default: 1).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Worker processes for each sweep (default: 16).",
    )
    parser.add_argument(
        "--executor",
        choices=("thread", "process"),
        default="process",
        help="Parallel executor backend (default: process).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    parser.add_argument(
        "--style",
        default="style/neuroips.mplstyle",
        help="Matplotlib style name or path.",
    )
    parser.add_argument(
        "--raster-plots",
        choices=["none", "single", "all"],
        default="single",
        help=(
            "Raster-plot mode passed to sweep script: "
            "'none' = no rasters; "
            "'single' = one raster per sweep point (default); "
            "'all' = one raster per repeat."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip configs whose output already contains sweep_summary.parquet.",
    )
    parser.add_argument(
        "--config-workers",
        type=int,
        default=1,
        help="How many configs to run in parallel (default: 1, sequential).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--M",
        nargs="+",
        type=int,
        default=None,
        metavar="M_VALUE",
        help="Restrict to specific M values, e.g. --M 4 5 6.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    config_dir = ROOT / args.config_dir
    save_root = ROOT / args.save_dir
    sweep_low, sweep_high, sweep_num = args.range
    sweep_num = int(sweep_num)

    # Discover and sort configs
    all_configs = sorted(config_dir.glob("snn_validate_*.yaml"), key=_sort_key)
    if not all_configs:
        print(f"No configs found in {config_dir}", file=sys.stderr)
        return 1

    # Optional M filter
    if args.M is not None:
        allowed = {f"M{m}" for m in args.M}
        all_configs = [
            c for c in all_configs
            if (parts := _parse_config_name(c)) is not None and parts[0] in allowed
        ]
        if not all_configs:
            print(f"No configs matched M filter {args.M}", file=sys.stderr)
            return 1

    total = len(all_configs)
    n_skipped = n_done = n_failed = 0

    def _run_one(i: int, config_path: Path) -> str:
        """Run a single config. Returns 'skipped' | 'done' | 'failed'."""
        parts = _parse_config_name(config_path)
        if parts is None:
            return "skipped"
        M_label, attractor_label = parts
        save_dir = save_root / M_label / attractor_label
        prefix = f"[{i}/{total}] {M_label}/{attractor_label}"

        if args.skip_existing and _is_done(save_dir):
            print(f"[skip] {prefix}", flush=True)
            return "skipped"

        cmd = _build_cmd(
            config_path,
            save_dir,
            sweep_low=sweep_low,
            sweep_high=sweep_high,
            sweep_num=sweep_num,
            n_repeats=args.n_repeats,
            max_workers=args.max_workers,
            executor=args.executor,
            log_level=args.log_level,
            style=args.style,
            raster_plots=args.raster_plots,
        )

        if args.dry_run:
            print(f"{prefix}  →  {' '.join(str(c) for c in cmd)}", flush=True)
            return "done"

        print(f"{prefix}", flush=True)
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            print(f"  WARNING: sweep failed (exit {result.returncode})", flush=True)
            return "failed"
        return "done"

    def _tally(outcome: str) -> None:
        nonlocal n_skipped, n_done, n_failed
        if outcome == "skipped":
            n_skipped += 1
        elif outcome == "failed":
            n_failed += 1
        else:
            n_done += 1

    if args.config_workers == 1:
        for i, config_path in enumerate(all_configs, start=1):
            _tally(_run_one(i, config_path))
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.config_workers) as pool:
            futs = {
                pool.submit(_run_one, i, cfg): cfg
                for i, cfg in enumerate(all_configs, start=1)
            }
            for fut in concurrent.futures.as_completed(futs):
                _tally(fut.result())

    if not args.dry_run:
        print(
            f"\nDone. completed={n_done}  skipped={n_skipped}  failed={n_failed}"
            f"  total={total}"
        )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
