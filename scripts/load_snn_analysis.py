#!/usr/bin/env python
"""Load and inspect SNN analysis data.

This utility script loads processed SNN analysis data and provides
a summary of the attractors and transition matrix. Useful for quick
inspection of experiment results.

Usage:
    python scripts/load_snn_analysis.py --analysis-dir simulations/snn_long_run/analysis

    # Export to DataFrame
    python scripts/load_snn_analysis.py --analysis-dir simulations/snn_long_run/analysis --export
"""

from __future__ import annotations

import argparse
from pathlib import Path

from neuro_mod.core.spiking_net.analysis import SNNAnalyzer
from neuro_mod.execution.helpers.logger import Logger


def _build_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load and inspect SNN analysis data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=root / "simulations" / "snn_long_run" / "analysis",
        help="Directory containing processed analysis data.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export the data to a pandas DataFrame and print summary.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed attractor information.",
    )
    return parser


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    args = _build_parser(root).parse_args()

    logger = Logger(name="LoadSNNAnalysis")
    analysis_dir = args.analysis_dir

    if not analysis_dir.exists():
        logger.error(f"Analysis directory not found: {analysis_dir}")
        return 1

    logger.info(f"Loading analysis from {analysis_dir}")
    analyzer = SNNAnalyzer(analysis_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("SNN ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Number of attractors: {analyzer.get_num_states()}")
    print(f"Transition matrix shape: {analyzer.get_transition_matrix().shape}")
    print(f"Total duration: {analyzer.total_duration_ms:.2f} ms")
    print(f"Time step (dt): {analyzer.dt} s")
    if analyzer.repeat_durations_ms:
        repeat_durations = ", ".join(f"{d:.2f}" for d in analyzer.repeat_durations_ms)
        print(f"Repeat durations: [{repeat_durations}] ms")
    if analyzer.n_runs is not None:
        print(f"Number of runs: {analyzer.n_runs}")

    # Get summary metrics
    metrics = analyzer.get_summary_metrics()
    print("\nSummary Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    if args.verbose:
        # Print top attractors
        attractors = analyzer.get_attractors_data()
        occurrences = analyzer.get_occurrences()
        mean_lifespans, _ = analyzer.get_life_spans()

        print("\nTop 10 Attractors by Occurrences:")
        print("-" * 50)
        sorted_indices = occurrences.argsort()[::-1][:10]
        for i, idx in enumerate(sorted_indices):
            print(
                f"  {i+1}. Attractor {idx}: "
                f"{int(occurrences[idx])} occurrences, "
                f"{mean_lifespans[idx]:.2f} ms mean lifespan"
            )

    if args.export:
        # Export to DataFrame
        df = analyzer.to_dataframe()
        print(f"\nDataFrame exported: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 10 rows:")
        print(df.head(10).to_string())

        # Save to CSV
        csv_path = analysis_dir / "exported_attractors.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nExported to: {csv_path}")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
